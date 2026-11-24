#include "server-dash.h"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

static const std::map<std::string, std::string> k_dash_mime = {
    { ".html",  "text/html; charset=utf-8"       },
    { ".css",   "text/css; charset=utf-8"        },
    { ".js",    "text/javascript; charset=utf-8" },
    { ".mjs",   "text/javascript; charset=utf-8" },
    { ".json",  "application/json"               },
    { ".svg",   "image/svg+xml"                  },
    { ".png",   "image/png"                      },
    { ".ico",   "image/x-icon"                   },
    { ".woff2", "font/woff2"                     },
    { ".txt",   "text/plain; charset=utf-8"      },
};

// the page loads only its own files and talks only to this server; nothing inline
static const char * k_dash_csp =
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

static bool dash_segment_ok(const std::string & s) {
    if (s.empty() || s[0] == '.') {
        return false;
    }
    for (const unsigned char c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
        if (!ok) {
            return false;
        }
    }
    return true;
}

// a 64-bit FNV-1a of the bytes and their length: the tag only has to change when the bytes do
static std::string dash_etag(const std::string & body) {
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char c : body) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "\"%016llx-%llx\"", (unsigned long long) h, (unsigned long long) body.size());
    return buf;
}

server_dash_assets::server_dash_assets(const std::string & root_path) {
    std::error_code ec;
    const fs::path p = fs::canonical(fs::path(root_path), ec);
    if (ec || !fs::is_directory(p, ec)) {
        throw std::runtime_error("dashboard path is not a directory: " + root_path);
    }
    root = fs::path(root_path);
}

std::optional<server_dash_asset> server_dash_assets::get(const std::string & rel_in) const {
    const std::string rel = rel_in.empty() ? std::string("index.html") : rel_in;

    std::error_code ec;
    const fs::path base = fs::canonical(root, ec);
    if (ec) {
        return std::nullopt;
    }

    // every segment from the allowed set: this alone keeps "..", "/", "\", NUL and percent escapes out
    fs::path p = base;
    size_t start = 0;
    while (true) {
        const size_t end = rel.find('/', start);
        const std::string seg = rel.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!dash_segment_ok(seg)) {
            return std::nullopt;
        }
        p /= seg;
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    const auto mime = k_dash_mime.find(p.extension().string());
    if (mime == k_dash_mime.end()) {
        return std::nullopt;
    }

    // symlinks resolve; the target must still be inside the root
    const fs::path real = fs::canonical(p, ec);
    if (ec) {
        return std::nullopt;
    }
    {
        auto r = base.begin();
        auto q = real.begin();
        for (; r != base.end(); ++r, ++q) {
            if (q == real.end() || *q != *r) {
                return std::nullopt;
            }
        }
    }
    if (!fs::is_regular_file(real, ec) || ec) {
        return std::nullopt;
    }
    const auto mtime = fs::last_write_time(real, ec);
    if (ec) {
        return std::nullopt;
    }
    const uintmax_t size = fs::file_size(real, ec);
    if (ec) {
        return std::nullopt;
    }

    const std::string key = real.string();
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = cache.find(key);
        if (it != cache.end() && it->second.mtime == mtime && it->second.size == size) {
            return it->second.asset;
        }
    }

    std::ifstream f(real, std::ios::binary);
    if (!f) {
        return std::nullopt;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    auto body = std::make_shared<const std::string>(ss.str());

    server_dash_asset a;
    a.etag         = dash_etag(*body);
    a.body         = std::move(body);
    a.content_type = mime->second;

    std::lock_guard<std::mutex> lock(mutex);
    cache[key] = entry { mtime, size, a };
    return a;
}

// If-None-Match: a list of tags (weak ones compare by their opaque part), or "*"
static bool dash_etag_matches(const std::string & header, const std::string & etag) {
    size_t start = 0;
    while (start <= header.size()) {
        size_t end = header.find(',', start);
        if (end == std::string::npos) {
            end = header.size();
        }
        std::string t = header.substr(start, end - start);
        const size_t a = t.find_first_not_of(" \t");
        const size_t b = t.find_last_not_of(" \t");
        t = a == std::string::npos ? std::string() : t.substr(a, b - a + 1);
        if (t.rfind("W/", 0) == 0) {
            t = t.substr(2);
        }
        if (t == "*" || t == etag) {
            return true;
        }
        start = end + 1;
    }
    return false;
}

static std::string string_lower(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

static void dash_security_headers(server_http_res & res) {
    res.headers["Content-Security-Policy"] = k_dash_csp;
    res.headers["X-Content-Type-Options"]  = "nosniff";
    res.headers["Referrer-Policy"]         = "no-referrer";
    res.headers["X-Frame-Options"]         = "DENY";
}

static server_http_res_ptr dash_not_found() {
    auto res = std::make_unique<server_http_res>();
    res->status       = 404;
    res->content_type = "text/plain; charset=utf-8";
    res->data         = "Not Found\n";
    dash_security_headers(*res);
    return res;
}

server_http_context::handler_t server_dash_handler(std::shared_ptr<const server_dash_assets> assets, std::string api_prefix) {
    const std::string mount = api_prefix + "/dash/";
    return [assets = std::move(assets), mount](const server_http_req & req) -> server_http_res_ptr {
        if (req.path.compare(0, mount.size(), mount) != 0) {
            return dash_not_found();
        }
        const auto a = assets->get(req.path.substr(mount.size()));
        if (!a) {
            return dash_not_found();
        }

        auto res = std::make_unique<server_http_res>();
        dash_security_headers(*res);
        res->headers["ETag"]          = a->etag;
        res->headers["Cache-Control"] = "no-cache";

        for (const auto & [k, v] : req.headers) {
            if (string_lower(k) == "if-none-match" && dash_etag_matches(v, a->etag)) {
                res->status = 304;
                return res;
            }
        }
        res->status       = 200;
        res->content_type = a->content_type;
        res->data         = *a->body;
        return res;
    };
}

server_http_context::handler_t server_dash_redirect(std::string api_prefix) {
    const std::string location = api_prefix + "/dash/";
    return [location](const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        res->status              = 308;
        res->content_type        = "text/plain; charset=utf-8";
        res->headers["Location"] = location;
        return res;
    };
}

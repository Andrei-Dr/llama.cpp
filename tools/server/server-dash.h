#pragma once

// GET /dash/*: the statistics dashboard's static files (index.html, css/, js/) from --dash-path.
//
// The HTTP thread reads a file on request (a deploy is live at once, a switched symlink too) and keeps its bytes while its mtime and size hold.
// Every file has a strong ETag of its bytes; the page revalidates each load (Cache-Control: no-cache) and an unchanged
// file costs a 304. Nothing here touches the server's main loop.

#include "server-http.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

struct server_dash_asset {
    std::shared_ptr<const std::string> body;
    std::string etag;          // quoted, strong
    std::string content_type;
};

class server_dash_assets {
public:
    // throws std::runtime_error when root is not a directory
    explicit server_dash_assets(const std::string & root);

    // rel: the path below /dash/ (percent-decoded once by the HTTP layer); empty = index.html. nullopt for anything that
    // is not a regular file of a known type inside the root: a segment that is empty, starts with '.' or has a character
    // outside [A-Za-z0-9_.-] (so no "..", no backslash, no NUL, no second decoding), or a symlink that resolves outside
    std::optional<server_dash_asset> get(const std::string & rel) const;

private:
    std::filesystem::path root;  // as configured: resolved again by every request (a deploy may switch a symlink)

    struct entry {
        std::filesystem::file_time_type mtime;
        uintmax_t                       size = 0;
        server_dash_asset               asset;
    };
    mutable std::mutex                   mutex;
    mutable std::map<std::string, entry> cache; // by canonical path
};

// the GET handler of api_prefix + "/dash/(.*)"
server_http_context::handler_t server_dash_handler(std::shared_ptr<const server_dash_assets> assets, std::string api_prefix);

// the GET handler of api_prefix + "/dash": a redirect to the directory, so the page's relative URLs resolve under it
server_http_context::handler_t server_dash_redirect(std::string api_prefix);

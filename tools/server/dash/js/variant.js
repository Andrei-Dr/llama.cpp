// layout variants under review (?v=a|b|c); removed once one is chosen
const v = new URLSearchParams(location.search).get("v") || "a";
document.documentElement.dataset.v = v;
for (const a of document.querySelectorAll("#vsw a")) if (a.getAttribute("href") === "?v=" + v) a.className = "on";

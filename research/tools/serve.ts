// Static server for the demo: bun run research/tools/serve.ts  ->  http://localhost:8787/demo/
const root = new URL("../..", import.meta.url).pathname;
const server = Bun.serve({
  port: +(process.env.PORT ?? 8787),
  async fetch(req) {
    const path = decodeURIComponent(new URL(req.url).pathname);
    if (path.includes("..")) return new Response("bad path", { status: 400 });
    const file = Bun.file(root + (path.endsWith("/") ? path + "index.html" : path));
    return (await file.exists()) ? new Response(file) : new Response("not found", { status: 404 });
  },
});
console.log(`demo: http://localhost:${server.port}/demo/`);

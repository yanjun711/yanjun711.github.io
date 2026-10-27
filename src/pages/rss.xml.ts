import { getCollection } from 'astro:content';
export async function GET() {
  const escape = (s: string) => s.replace(/[<>&'\"]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', "'": '&apos;', '"': '&quot;' }[c]!));
  const posts = (await getCollection('blog', ({data}) => !data.draft && data.date <= new Date())).sort((a,b) => b.data.date.valueOf()-a.data.date.valueOf());
  return new Response(`<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>Yanjun Li — Blogs</title><link>https://yanjun711.github.io/blog/</link><description>Research notes by Yanjun Li.</description><language>en</language>${posts.map(post => `<item><title>${escape(post.data.title)}</title><link>https://yanjun711.github.io/blog/${post.id}/</link><guid>https://yanjun711.github.io/blog/${post.id}/</guid><description>${escape(post.data.description)}</description><pubDate>${post.data.date.toUTCString()}</pubDate></item>`).join('')}</channel></rss>`, { headers: { 'Content-Type': 'application/rss+xml; charset=utf-8' } });
}

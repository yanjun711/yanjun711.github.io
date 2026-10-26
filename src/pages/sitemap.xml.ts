import { getCollection } from 'astro:content';
export async function GET() {
  const posts = await getCollection('blog', ({data}) => !data.draft && data.date <= new Date());
  const paths = ['/', '/blog/', ...posts.map(post => `/blog/${post.id}/`)];
  return new Response(`<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${paths.map(path => `<url><loc>https://yanjun711.github.io${path}</loc></url>`).join('')}</urlset>`, { headers: { 'Content-Type': 'application/xml; charset=utf-8' } });
}

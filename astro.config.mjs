import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { unified } from '@astrojs/markdown-remark';

// Keep code and long equations reachable when they scroll on small screens.
function focusableBlocks() {
  return (tree) => {
    function walk(node) {
      if (node.type === 'element' && (node.tagName === 'pre' || node.properties?.className?.includes('katex-display'))) {
        node.properties ??= {};
        node.properties.tabIndex = 0;
      }
      node.children?.forEach(walk);
    }
    walk(tree);
  };
}

export default defineConfig({
  site: 'https://yanjun711.github.io',
  trailingSlash: 'always',
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, { strict: 'error', throwOnError: true }], focusableBlocks],
    }),
    shikiConfig: { theme: 'github-light-high-contrast' },
  },
});

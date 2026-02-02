import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import wickGrammar from '../../editors/textmate/wick.tmLanguage.json'

export default withMermaid(
  defineConfig({
    vite: {
      optimizeDeps: {
        include: ['mermaid'],
        exclude: ['wick_wasm'],
      },
      build: {
        rollupOptions: {
          output: {
            manualChunks(id) {
              if (id.includes('monaco-editor')) return 'monaco'
              if (id.includes('playground')) return 'playground'
            },
          },
        },
      },
    },
    markdown: {
      languages: [wickGrammar as any],
    },
    title: 'Wick',
    description: 'Minimal expression language with multiple backends',

    base: '/wick/',

    themeConfig: {
      nav: [
        { text: 'Guide', link: '/introduction' },
        { text: 'Backends', link: '/backends/wgsl' },
        { text: 'Playground', link: '/playground' },
        { text: 'rhi', link: 'https://docs.rhi.zone/' },
      ],

      sidebar: {
        '/': [
          {
            text: 'Guide',
            items: [
              { text: 'Introduction', link: '/introduction' },
              { text: 'Use Cases', link: '/use-cases' },
              { text: 'Integration', link: '/integration' },
              { text: 'Optimization', link: '/optimization' },
            ]
          },
          {
            text: 'Crates',
            items: [
              { text: 'wick-core', link: '/core' },
              { text: 'wick-scalar', link: '/scalar' },
              { text: 'wick-linalg', link: '/linalg' },
              { text: 'wick-complex', link: '/complex' },
              { text: 'wick-quaternion', link: '/quaternion' },
            ]
          },
          {
            text: 'Backends',
            items: [
              { text: 'WGSL', link: '/backends/wgsl' },
              { text: 'GLSL', link: '/backends/glsl' },
              { text: 'OpenCL', link: '/backends/opencl' },
              { text: 'CUDA', link: '/backends/cuda' },
              { text: 'HIP', link: '/backends/hip' },
              { text: 'Rust', link: '/backends/rust' },
              { text: 'C', link: '/backends/c' },
              { text: 'TokenStream', link: '/backends/tokenstream' },
              { text: 'Lua', link: '/backends/lua' },
              { text: 'Cranelift', link: '/backends/cranelift' },
            ]
          },
          {
            text: 'Reference',
            items: [
              { text: 'API Reference', link: '/api-reference' },
            ]
          },
        ]
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/rhi-zone/wick' }
      ],

      search: {
        provider: 'local'
      },

      editLink: {
        pattern: 'https://github.com/rhi-zone/wick/edit/master/docs/:path',
        text: 'Edit this page on GitHub'
      },
    },
  }),
)

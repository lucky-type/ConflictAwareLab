import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        notion: {
          bg: '#FFFFFF',
          sidebar: '#F7F6F3',
          hover: '#EFEFEF',
          'light-gray': '#F1F1EF',
          text: '#37352F',
          'text-secondary': '#787774',
          'text-tertiary': '#9B9A97',
          border: '#E3E2E0',
          'border-dark': '#D3D3D3',
          blue: '#337EA9',
          green: '#448361',
          red: '#D44C47',
          orange: '#CB7B37',
          purple: '#9065B0',
          pink: '#C14C8A',
          yellow: '#C29243',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
      },
    },
  },
  plugins: [],
} satisfies Config;

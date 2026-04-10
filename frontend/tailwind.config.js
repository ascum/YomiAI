/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans:  ["DM Sans", "system-ui", "sans-serif"],
        serif: ["Playfair Display", "Georgia", "serif"],
        mono:  ["DM Mono", "ui-monospace", "monospace"],
      },
      colors: {
        k: {
          cream:  "#fffef7",
          navy:   "#2e3257",
          muted:  "#627d9a",
          gray:   "#babbbd",
          accent: "#dfc5a4",
        },
      },
    },
  },
  plugins: [],
}
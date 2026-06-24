/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#05070b",
        foreground: "#f5f7fb",
        muted: "#93a4b8",
        border: "rgba(148, 163, 184, 0.24)",
      },
      boxShadow: {
        panel: "0 18px 60px rgba(0, 0, 0, 0.35)",
      },
    },
  },
  plugins: [],
};

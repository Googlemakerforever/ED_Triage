import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./features/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Manrope", "Avenir Next", "Segoe UI", "ui-sans-serif", "system-ui", "sans-serif"]
      },
      colors: {
        brand: {
          50: "#f0f7ff",
          100: "#dcecff",
          500: "#1f6feb",
          700: "#1246a6",
          900: "#0a2459"
        }
      },
      boxShadow: {
        soft: "0 10px 30px rgba(12, 28, 61, 0.08)"
      }
    }
  },
  plugins: []
};

export default config;

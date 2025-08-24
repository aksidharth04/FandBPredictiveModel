/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        honeywell: {
          primary: '#D32F2F',
          secondary: '#1976D2',
          accent: '#FFC107',
          success: '#388E3C',
          warning: '#F57C00',
          danger: '#D32F2F',
          'light-gray': '#F5F5F5',
          'dark-gray': '#424242'
        }
      },
      fontFamily: {
        'helvetica': ['Helvetica', 'Arial', 'sans-serif']
      }
    },
  },
  plugins: [],
}

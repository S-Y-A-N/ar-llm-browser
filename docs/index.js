const themeBtn = document.querySelector(".theme-btn");
themeBtn.addEventListener("click", () => toggleTheme());

window.addEventListener("DOMContentLoaded", () => {
  const theme = localStorage.getItem("theme");
  if (theme) {
    document.documentElement.dataset["theme"] = theme
    if (theme === "dark") {
      document.getElementById("light-icon").classList.toggle("hidden")
      document.getElementById("dark-icon").classList.toggle("hidden")
    }
  } else {
    const systemDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
    document.documentElement.dataset["theme"] = systemDark.matches ? "dark" : "light"
    document.getElementById("light-icon").classList.toggle("hidden")
    document.getElementById("dark-icon").classList.toggle("hidden")
  }
});

function toggleTheme() {
  if (document.documentElement.dataset["theme"] === "light") {
    document.documentElement.dataset["theme"] = "dark"
    localStorage.setItem("theme", "dark")
  } else {
    document.documentElement.dataset["theme"] = "light"
    localStorage.setItem("theme", "light")
  }

  document.getElementById("light-icon").classList.toggle("hidden")
  document.getElementById("dark-icon").classList.toggle("hidden")
}
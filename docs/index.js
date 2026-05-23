/* ===== Theme toggle (preserves original localStorage behavior) ===== */
const themeBtn = document.querySelector(".theme-btn");
themeBtn.addEventListener("click", () => toggleTheme());

window.addEventListener("DOMContentLoaded", () => {
  const theme = localStorage.getItem("theme");
  if (theme) {
    document.documentElement.dataset["theme"] = theme;
    if (theme === "dark") {
      document.getElementById("light-icon").classList.toggle("hidden");
      document.getElementById("dark-icon").classList.toggle("hidden");
    }
  } else {
    const systemDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
    document.documentElement.dataset["theme"] = systemDark.matches ? "dark" : "light";
    document.getElementById("light-icon").classList.toggle("hidden");
    document.getElementById("dark-icon").classList.toggle("hidden");
  }
});

function toggleTheme() {
  if (document.documentElement.dataset["theme"] === "light") {
    document.documentElement.dataset["theme"] = "dark";
    localStorage.setItem("theme", "dark");
  } else {
    document.documentElement.dataset["theme"] = "light";
    localStorage.setItem("theme", "light");
  }
  document.getElementById("light-icon").classList.toggle("hidden");
  document.getElementById("dark-icon").classList.toggle("hidden");
}

/* ===== Language toggle (English ⇄ Arabic, with RTL) ===== */
const langBtn = document.querySelector(".lang-btn");
const langBtnText = document.querySelector(".lang-btn-text");

function applyLang(lang) {
  const root = document.documentElement;
  root.dataset.lang = lang;
  root.setAttribute("lang", lang === "ar" ? "ar" : "en");
  root.setAttribute("dir", lang === "ar" ? "rtl" : "ltr");
  if (langBtnText) langBtnText.textContent = lang === "ar" ? "English" : "العربية";
}

(function initLang() {
  const saved = localStorage.getItem("lang");
  applyLang(saved === "ar" ? "ar" : "en");
})();

if (langBtn) {
  langBtn.addEventListener("click", () => {
    const next = document.documentElement.dataset.lang === "ar" ? "en" : "ar";
    localStorage.setItem("lang", next);
    applyLang(next);
  });
}

/* ===== Smooth scroll with sticky-nav offset ===== */
function scrollToId(id) {
  const node = document.getElementById(id);
  if (!node) return;
  const nav = document.querySelector("nav.topnav");
  const navH = nav ? nav.offsetHeight : 0;
  const top = node.getBoundingClientRect().top + window.pageYOffset - navH - 14;
  window.scrollTo({ top: Math.max(top, 0), left: 0, behavior: "smooth" });
}

document.querySelectorAll('a[href^="#"]').forEach((link) => {
  link.addEventListener("click", (e) => {
    const id = link.getAttribute("href").slice(1);
    if (!id) return;
    const target = document.getElementById(id);
    if (!target) return;
    e.preventDefault();
    scrollToId(id);
    history.replaceState(null, "", `#${id}`);
  });
});

/* ===== Nav: shrink / glass on scroll ===== */
const topnav = document.getElementById("topnav");
const onScroll = () => {
  if (window.scrollY > 40) topnav.classList.add("scrolled");
  else topnav.classList.remove("scrolled");
};
window.addEventListener("scroll", onScroll, { passive: true });
onScroll();

/* ===== Reveal sections on scroll ===== */
const revealEls = document.querySelectorAll(".reveal");
const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("in");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
);
revealEls.forEach((el) => revealObserver.observe(el));

/* ===== Scroll-spy: highlight active nav link ===== */
const sections = document.querySelectorAll("main section[id]");
const navLinks = document.querySelectorAll("nav.topnav a.nav-link");

const spyObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        navLinks.forEach((l) =>
          l.classList.toggle("active", l.getAttribute("href") === `#${id}`)
        );
      }
    });
  },
  { rootMargin: "-45% 0px -50% 0px", threshold: 0 }
);
sections.forEach((s) => spyObserver.observe(s));

/* ===== Animated counters for key results ===== */
function animateCount(el) {
  const target = parseFloat(el.dataset.count);
  if (isNaN(target)) return;
  const prefix = el.dataset.prefix ? el.dataset.prefix.replace("&gt;", ">") : "";
  const suffix = el.dataset.suffix || "";
  const dur = 1400;
  const start = performance.now();
  function tick(now) {
    const p = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = prefix + Math.round(target * eased) + suffix;
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

const statObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting && entry.target.dataset.count) {
        animateCount(entry.target);
        statObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.6 }
);
document.querySelectorAll(".stat .num[data-count]").forEach((el) => statObserver.observe(el));

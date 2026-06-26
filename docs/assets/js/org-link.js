// Point the top-left org logo/brand at the GitHub organization.
(function () {
  var ORG_URL = "https://github.com/Cloud2BR-MSFTLearningHub";
  var HOME_URL = "/Azure-ML-Advanced/";
  function pointLogoToOrg() {
    document
      .querySelectorAll('a.md-logo, a[data-md-component="logo"], a.md-header__button.md-logo')
      .forEach(function (logo) {
        logo.setAttribute("href", ORG_URL);
        logo.setAttribute("target", "_blank");
        logo.setAttribute("rel", "noopener");
      });
  }
  function pointTitleToHome() {
    document.querySelectorAll(".md-header__title .md-ellipsis").forEach(function (title) {
      title.style.cursor = "pointer";
      title.addEventListener("click", function () {
        window.location.href = HOME_URL;
      });
    });
  }
  function init() {
    pointLogoToOrg();
    pointTitleToHome();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

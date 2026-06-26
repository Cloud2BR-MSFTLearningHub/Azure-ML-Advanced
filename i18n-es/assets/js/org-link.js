// Point the top-left org logo/brand at the GitHub organization.
(function () {
  var ORG_URL = "https://github.com/Cloud2BR-MSFTLearningHub";
  function pointLogoToOrg() {
    document
      .querySelectorAll('a.md-logo, a[data-md-component="logo"], a.md-header__button.md-logo')
      .forEach(function (logo) {
        logo.setAttribute("href", ORG_URL);
        logo.setAttribute("target", "_blank");
        logo.setAttribute("rel", "noopener");
      });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", pointLogoToOrg);
  } else {
    pointLogoToOrg();
  }
})();

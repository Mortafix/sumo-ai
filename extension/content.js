(function initSumoForYouTube(globalScope) {
  "use strict";

  const SUMO_BASE_URL = "https://sumo.moris.dev/";
  const YOUTUBE_ORIGIN = "https://www.youtube.com";
  const VIDEO_ID_PATTERN = /^[A-Za-z0-9_-]{11}$/;
  const THUMBNAIL_BUTTON_CLASS = "sumo-thumbnail-button";
  const THUMBNAIL_HOST_CLASS = "sumo-thumbnail-host";
  const HOME_CARD_CLASS = "sumo-home-video-card";
  const INFO_MOUNT_CLASS = "sumo-info-button-host";
  const WATCH_BUTTON_ID = "sumo-watch-button";
  const VIDEO_CARD_SELECTOR = [
    "ytd-rich-grid-media",
    "ytd-video-renderer",
    "ytd-grid-video-renderer",
    "ytd-compact-video-renderer",
    "ytd-playlist-video-renderer",
    "yt-lockup-view-model"
  ].join(",");
  const THUMBNAIL_LINK_SELECTOR = [
    'a#thumbnail[href*="/watch"]',
    'yt-lockup-view-model a[href*="/watch?v="]',
    'yt-lockup-view-model a[href^="/live/"]'
  ].join(",");

  function isYouTubeHostname(hostname) {
    const normalized = (hostname || "").toLowerCase();
    return (
      normalized === "youtu.be" ||
      normalized === "youtube.com" ||
      normalized.endsWith(".youtube.com")
    );
  }

  function extractVideoId(value) {
    if (typeof value !== "string" || !value.trim()) {
      return null;
    }

    let parsed;
    try {
      parsed = new URL(value, YOUTUBE_ORIGIN);
    } catch (_error) {
      return null;
    }

    if (!isYouTubeHostname(parsed.hostname)) {
      return null;
    }

    const pathParts = parsed.pathname.split("/").filter(Boolean);
    if (parsed.hostname.toLowerCase() === "youtu.be") {
      return VIDEO_ID_PATTERN.test(pathParts[0] || "") ? pathParts[0] : null;
    }

    if (parsed.pathname === "/watch") {
      const watchId = parsed.searchParams.get("v") || "";
      return VIDEO_ID_PATTERN.test(watchId) ? watchId : null;
    }

    if (pathParts[0] === "live" && VIDEO_ID_PATTERN.test(pathParts[1] || "")) {
      return pathParts[1];
    }

    return null;
  }

  function canonicalizeYouTubeUrl(value) {
    const videoId = extractVideoId(value);
    return videoId ? `${YOUTUBE_ORIGIN}/watch?v=${videoId}` : null;
  }

  function buildSumoUrl(value) {
    const canonicalUrl = canonicalizeYouTubeUrl(value);
    if (!canonicalUrl) {
      return null;
    }

    const target = new URL(SUMO_BASE_URL);
    target.searchParams.set("url", canonicalUrl);
    return target.toString();
  }

  function stopYouTubeNavigation(event) {
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
  }

  function openInSumo(event) {
    stopYouTubeNavigation(event);
    const sourceUrl = event.currentTarget.dataset.sumoVideoUrl;
    const targetUrl = buildSumoUrl(sourceUrl);
    if (targetUrl) {
      globalScope.open(targetUrl, "_blank", "noopener,noreferrer");
    }
  }

  function createSparklesIcon() {
    const svgNamespace = "http://www.w3.org/2000/svg";
    const icon = document.createElementNS(svgNamespace, "svg");
    icon.classList.add("sumo-button-icon");
    icon.setAttribute("viewBox", "0 0 24 24");
    icon.setAttribute("fill", "none");
    icon.setAttribute("aria-hidden", "true");

    const mainSparkle = document.createElementNS(svgNamespace, "path");
    mainSparkle.setAttribute(
      "d",
      "M12 2.75C12.55 7.2 14.8 9.45 19.25 10C14.8 10.55 12.55 12.8 12 17.25C11.45 12.8 9.2 10.55 4.75 10C9.2 9.45 11.45 7.2 12 2.75Z"
    );

    const smallSparkle = document.createElementNS(svgNamespace, "path");
    smallSparkle.setAttribute(
      "d",
      "M19 15.5C19.25 17.5 20.25 18.5 22.25 18.75C20.25 19 19.25 20 19 22C18.75 20 17.75 19 15.75 18.75C17.75 18.5 18.75 17.5 19 15.5Z"
    );

    icon.append(mainSparkle, smallSparkle);
    return icon;
  }

  function createThumbnailButton(videoUrl) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = THUMBNAIL_BUTTON_CLASS;
    button.dataset.sumoVideoUrl = videoUrl;
    button.setAttribute("aria-label", "Riassumi con Sumo");
    button.title = "Riassumi con Sumo";
    button.append(createSparklesIcon());

    const label = document.createElement("span");
    label.textContent = "Sumo";
    button.append(label);

    button.addEventListener("pointerdown", stopYouTubeNavigation);
    button.addEventListener("click", openInSumo);
    return button;
  }

  function findThumbnailHost(link) {
    const homeFeedCard = link.closest(
      "ytd-rich-grid-renderer ytd-rich-item-renderer"
    );
    if (homeFeedCard) {
      return homeFeedCard;
    }

    return (
      link.closest(VIDEO_CARD_SELECTOR) ||
      link.closest("ytd-thumbnail, yt-thumbnail-view-model") ||
      link.parentElement
    );
  }

  function isHomeFeedCard(host) {
    return (
      host.matches("ytd-rich-item-renderer, ytd-rich-grid-media") &&
      Boolean(host.closest("ytd-rich-grid-renderer"))
    );
  }

  function findButtonMount(host) {
    if (host.matches("yt-lockup-view-model")) {
      return (
        host.querySelector(".ytLockupMetadataViewModelTextContainer") || host
      );
    }

    if (
      host.matches(
        "ytd-video-renderer, ytd-compact-video-renderer, ytd-playlist-video-renderer"
      )
    ) {
      return (
        host.querySelector("#meta, #metadata, #details") || host
      );
    }

    return host;
  }

  function scanThumbnails() {
    document.querySelectorAll(THUMBNAIL_LINK_SELECTOR).forEach((link) => {
      const isVisualLink =
        link.id === "thumbnail" ||
        Boolean(link.querySelector("img, yt-image, .yt-core-image"));
      const videoUrl = canonicalizeYouTubeUrl(link.getAttribute("href") || "");
      if (
        !isVisualLink ||
        !videoUrl ||
        link.closest("ytd-reel-item-renderer")
      ) {
        return;
      }

      const host = findThumbnailHost(link);
      if (!host) {
        return;
      }

      host.classList.add(THUMBNAIL_HOST_CLASS);
      host.classList.toggle(HOME_CARD_CLASS, isHomeFeedCard(host));
      const buttonMount = findButtonMount(host);
      buttonMount.classList.toggle(INFO_MOUNT_CLASS, buttonMount !== host);

      host
        .querySelectorAll(`.${THUMBNAIL_BUTTON_CLASS}`)
        .forEach((button) => {
          if (button.parentElement !== buttonMount) {
            button.remove();
          }
        });

      const existingButton = Array.from(buttonMount.children).find((child) =>
        child.classList?.contains(THUMBNAIL_BUTTON_CLASS)
      );

      if (existingButton) {
        existingButton.dataset.sumoVideoUrl = videoUrl;
        return;
      }

      buttonMount.append(createThumbnailButton(videoUrl));
    });
  }

  function createWatchButton(videoUrl) {
    const button = document.createElement("button");
    button.id = WATCH_BUTTON_ID;
    button.type = "button";
    button.className = "sumo-watch-button";
    button.dataset.sumoVideoUrl = videoUrl;
    button.setAttribute("aria-label", "Riassumi questo video con Sumo");
    button.title = "Riassumi questo video con Sumo";
    button.append(createSparklesIcon());

    const label = document.createElement("span");
    label.textContent = "Sumo";
    button.append(label);

    button.addEventListener("pointerdown", stopYouTubeNavigation);
    button.addEventListener("click", openInSumo);
    return button;
  }

  function syncWatchButton() {
    const videoUrl = canonicalizeYouTubeUrl(globalScope.location.href);
    const existingButton = document.getElementById(WATCH_BUTTON_ID);

    if (!videoUrl) {
      existingButton?.remove();
      return;
    }

    if (existingButton) {
      existingButton.dataset.sumoVideoUrl = videoUrl;
      return;
    }

    const actions =
      document.querySelector("ytd-watch-metadata #top-level-buttons-computed") ||
      document.querySelector("#top-level-buttons-computed");
    if (actions) {
      actions.append(createWatchButton(videoUrl));
    }
  }

  function bootstrap() {
    if (document.documentElement.dataset.sumoExtensionReady === "1") {
      return;
    }
    document.documentElement.dataset.sumoExtensionReady = "1";

    let scanScheduled = false;
    const scheduleScan = () => {
      if (scanScheduled) {
        return;
      }
      scanScheduled = true;
      globalScope.requestAnimationFrame(() => {
        scanScheduled = false;
        scanThumbnails();
        syncWatchButton();
      });
    };

    const observer = new MutationObserver(scheduleScan);
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true
    });

    document.addEventListener("yt-navigate-finish", scheduleScan);
    globalScope.addEventListener("popstate", scheduleScan);
    scheduleScan();
  }

  const publicApi = {
    SUMO_BASE_URL,
    extractVideoId,
    canonicalizeYouTubeUrl,
    buildSumoUrl
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = publicApi;
  } else if (globalScope && globalScope.document) {
    bootstrap();
  }
})(typeof window !== "undefined" ? window : globalThis);

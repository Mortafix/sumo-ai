const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const projectRoot = path.resolve(__dirname, "..");
const extensionRoot = path.join(projectRoot, "extension");
const {
  SUMO_BASE_URL,
  extractVideoId,
  canonicalizeYouTubeUrl,
  buildSumoUrl
} = require(path.join(extensionRoot, "content.js"));

const VIDEO_ID = "dQw4w9WgXcQ";

test("extractVideoId accepts supported YouTube video URLs", () => {
  assert.equal(extractVideoId(`/watch?v=${VIDEO_ID}&list=abc`), VIDEO_ID);
  assert.equal(
    extractVideoId(`https://www.youtube.com/watch?v=${VIDEO_ID}&t=30`),
    VIDEO_ID
  );
  assert.equal(extractVideoId(`https://youtu.be/${VIDEO_ID}?si=abc`), VIDEO_ID);
  assert.equal(extractVideoId(`https://www.youtube.com/live/${VIDEO_ID}`), VIDEO_ID);
});

test("extractVideoId rejects Shorts, foreign hosts and malformed IDs", () => {
  assert.equal(
    extractVideoId(`https://www.youtube.com/shorts/${VIDEO_ID}`),
    null
  );
  assert.equal(
    extractVideoId(`https://example.com/watch?v=${VIDEO_ID}`),
    null
  );
  assert.equal(extractVideoId("https://www.youtube.com/watch?v=too-short"), null);
});

test("canonicalizeYouTubeUrl strips playlists, timestamps and tracking", () => {
  assert.equal(
    canonicalizeYouTubeUrl(
      `https://www.youtube.com/watch?v=${VIDEO_ID}&list=PL123&t=45&si=tracking`
    ),
    `https://www.youtube.com/watch?v=${VIDEO_ID}`
  );
});

test("buildSumoUrl only adds the canonical video URL", () => {
  const result = new URL(
    buildSumoUrl(`https://www.youtube.com/watch?v=${VIDEO_ID}&t=90`)
  );

  assert.equal(`${result.origin}/`, SUMO_BASE_URL);
  assert.equal(
    result.searchParams.get("url"),
    `https://www.youtube.com/watch?v=${VIDEO_ID}`
  );
  assert.deepEqual([...result.searchParams.keys()], ["url"]);
});

test("manifest is MV3 and requests only the YouTube content-script match", () => {
  const manifest = JSON.parse(
    fs.readFileSync(path.join(extensionRoot, "manifest.json"), "utf8")
  );

  assert.equal(manifest.manifest_version, 3);
  assert.equal(manifest.version, "1.1.1");
  assert.deepEqual(manifest.permissions, undefined);
  assert.deepEqual(manifest.host_permissions, undefined);
  assert.deepEqual(manifest.browser_specific_settings, {
    gecko: {
      id: "sumo-for-youtube@moris.local",
      strict_min_version: "140.0",
      data_collection_permissions: {
        required: ["browsingActivity"]
      }
    }
  });
  assert.deepEqual(manifest.content_scripts, [
    {
      matches: ["https://www.youtube.com/*"],
      css: ["styles.css"],
      js: ["content.js"],
      run_at: "document_idle"
    }
  ]);

  for (const iconPath of Object.values(manifest.icons)) {
    assert.equal(fs.existsSync(path.join(extensionRoot, iconPath)), true);
  }
});

test("button UI uses a vector icon instead of an image element", () => {
  const contentScript = fs.readFileSync(
    path.join(extensionRoot, "content.js"),
    "utf8"
  );

  assert.match(contentScript, /createSparklesIcon/);
  assert.match(contentScript, /ytd-rich-grid-media/);
  assert.match(contentScript, /ytd-rich-item-renderer/);
  assert.match(contentScript, /isHomeFeedCard/);
  assert.match(contentScript, /ytLockupMetadataViewModelTextContainer/);
  assert.match(contentScript, /findButtonMount/);
  assert.match(contentScript, /link\.closest\(VIDEO_CARD_SELECTOR\)/);
  assert.match(contentScript, /sumo-home-video-card/);
  assert.doesNotMatch(contentScript, /globalScope\.chrome/);
  assert.doesNotMatch(contentScript, /:has\(/);
  assert.doesNotMatch(contentScript, /createElement\(["']img["']\)/);
  assert.doesNotMatch(contentScript, /sumo-button-logo/);
});

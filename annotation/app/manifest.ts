import type { MetadataRoute } from "next";

// Web app manifest — served by Next at /manifest.webmanifest (distinct from the
// app's data manifest at /api/manifest). Makes the tool installable to the iPad
// home screen and launch full-screen. Colors mirror app/globals.css.
export default function manifest(): MetadataRoute.Manifest {
  return {
    id: "/",
    name: "Aviary Annotation",
    short_name: "Aviary",
    description: "Box and label bird detections.",
    start_url: "/",
    scope: "/",
    display: "standalone",
    orientation: "any",
    background_color: "#0a0a0a",
    theme_color: "#0a0a0a",
    icons: [
      { src: "/icons/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
      { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
      { src: "/icons/maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The annotation tool is a local single-user app; no remote image optimization.
  images: { unoptimized: true },
  // sharp is a native module used by the crop route — keep it out of the server
  // bundle trace so the prebuilt binary loads at runtime.
  serverExternalPackages: ["sharp"],
};

export default nextConfig;

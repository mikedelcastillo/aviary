import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The annotation tool is a local single-user app; no remote image optimization.
  images: { unoptimized: true },
};

export default nextConfig;

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  // Configure turbopack to treat better-sqlite3 as external (native Node module)
  turbopack: {
    resolveExtensions: [".ts", ".tsx", ".js", ".jsx"],
  },
  // Proxy backend API calls through Next.js so a single tunnel URL works
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: "http://localhost:8000/:path*",
      },
    ];
  },
  // Also keep webpack config for non-turbopack builds
  webpack: (config, { isServer }) => {
    if (isServer) {
      const externals = config.externals || [];
      config.externals = Array.isArray(externals)
        ? [...externals, "better-sqlite3"]
        : externals;
    }
    return config;
  },
};

export default nextConfig;

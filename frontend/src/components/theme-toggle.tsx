"use client";

import { useTheme } from "next-themes";
import { useSyncExternalStore } from "react";
import { Sun, Moon } from "lucide-react";
import { Button } from "@/components/ui/button";

const emptySubscribe = () => () => {};

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const mounted = useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false
  );

  if (!mounted) {
    return (
      <Button
        variant="ghost"
        size="icon"
        aria-label="Toggle theme"
        className="h-8 w-8 text-[var(--muted)]"
      >
        <span className="h-4 w-4" />
      </Button>
    );
  }

  const isDark = resolvedTheme === "dark";

  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="Toggle theme"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className="h-8 w-8 rounded-lg text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--tile-2)] border border-transparent hover:border-[var(--border)] transition-colors"
    >
      {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4 text-[var(--text)]" />}
    </Button>
  );
}

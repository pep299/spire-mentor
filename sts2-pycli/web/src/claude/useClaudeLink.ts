import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch, isStaticMode } from "../transport";
import {
  ClaudeLinkClient,
  consumePairingFragment,
  DEFAULT_CLAUDE_LINK_URL,
  PROTOCOL_VERSION,
  storeLinkToken,
  storeLinkUrl,
  storedLinkToken,
  storedLinkUrl,
  type CallResult,
  type LinkStatus,
  type ShowItem,
} from "./link";

export interface DockItem extends ShowItem {
  id: number;
  at: number;
}

/** Most recent first; a title shown again replaces its item (Claude iterating
 *  on one chart should not stack copies). */
export function upsertItem(items: DockItem[], item: ShowItem, id: number, cap = 20): DockItem[] {
  const rest = items.filter((it) => it.title !== item.title);
  return [{ ...item, id, at: Date.now() }, ...rest].slice(0, cap);
}

export interface ClaudeLinkHost {
  /** What is on screen, for `wb_context`. Read at request time. */
  context(): unknown;
  /** Claude changed the run: update the view as the page's own control would. */
  afterMutation(route: string, result: CallResult, request: unknown): Promise<void>;
}

export interface ClaudeLinkState {
  url: string | null;
  status: LinkStatus;
  items: DockItem[];
  /** Bumped on every new item, so the dock can pop open. */
  lastShownId: number;
  /** Turn on (asks for the pairing token if this origin has none) / off. */
  setEnabled(on: boolean): void;
  /** Ask for a (new) pairing token and reconnect with it. */
  pair(): void;
  dismiss(id: number): void;
  clear(): void;
}

/** The page's Claude link: owned by App (a Tools-menu row unmounts when the
 *  menu closes, so it can only render a toggle for this). */
export function useClaudeLink(host: ClaudeLinkHost, mode: string): ClaudeLinkState {
  const hostRef = useRef(host);
  hostRef.current = host;
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const [url, setUrl] = useState<string | null>(() => {
    try {
      consumePairingFragment();
    } catch {
      /* a broken pairing link must never take the workbench down */
    }
    return storedLinkUrl();
  });
  /** Bumped to reconnect with a new token. */
  const [generation, setGeneration] = useState(0);
  const [status, setStatus] = useState<LinkStatus>("off");
  const [items, setItems] = useState<DockItem[]>([]);
  const [lastShownId, setLastShownId] = useState(0);
  const nextId = useRef(1);
  const clientRef = useRef<ClaudeLinkClient | null>(null);

  useEffect(() => {
    const token = storedLinkToken();
    if (!url || !token) {
      setStatus("off");
      return;
    }
    const client = new ClaudeLinkClient({
      url,
      token,
      fetcher: apiFetch,
      onStatus: setStatus,
      hello: () => ({
        url: globalThis.location?.href ?? null,
        mode: modeRef.current,
        static: isStaticMode(),
        protocol: PROTOCOL_VERSION,
      }),
      host: {
        context: () => hostRef.current.context(),
        afterMutation: (route, result, request) => hostRef.current.afterMutation(route, result, request),
        show: (item) => {
          const id = nextId.current++;
          setItems((prev) => upsertItem(prev, item, id));
          setLastShownId(id);
        },
      },
    });
    clientRef.current = client;
    setStatus("waiting");
    client.start();
    return () => {
      client.stop();
      if (clientRef.current === client) clientRef.current = null;
    };
  }, [url, generation]);

  useEffect(() => {
    clientRef.current?.announce();
  }, [mode]);

  // A pairing link pasted onto an open tab changes only the hash — no reload,
  // so the mount-time check above never sees it.
  useEffect(() => {
    const onHash = () => {
      if (!consumePairingFragment()) return;
      setUrl(storedLinkUrl());
      setGeneration((g) => g + 1);
    };
    globalThis.addEventListener?.("hashchange", onHash);
    return () => globalThis.removeEventListener?.("hashchange", onHash);
  }, []);

  const askToken = useCallback((): boolean => {
    const entered = globalThis.prompt?.(
      "Claude link pairing token (Claude Code shows it: ask it for the workbench pairing token, or see wb_status)",
      "",
    );
    if (!entered?.trim()) return false;
    storeLinkToken(entered);
    return true;
  }, []);
  const setEnabled = useCallback((on: boolean) => {
    if (on && !storedLinkToken() && !askToken()) return;
    const next = on ? (storedLinkUrl() ?? DEFAULT_CLAUDE_LINK_URL) : null;
    storeLinkUrl(next);
    setUrl(next);
  }, [askToken]);
  const pair = useCallback(() => {
    if (!askToken()) return;
    const next = storedLinkUrl() ?? DEFAULT_CLAUDE_LINK_URL;
    storeLinkUrl(next);
    setUrl(next);
    setGeneration((g) => g + 1);
  }, [askToken]);
  const dismiss = useCallback((id: number) => setItems((prev) => prev.filter((it) => it.id !== id)), []);
  const clear = useCallback(() => setItems([]), []);

  return { url, status, items, lastShownId, setEnabled, pair, dismiss, clear };
}

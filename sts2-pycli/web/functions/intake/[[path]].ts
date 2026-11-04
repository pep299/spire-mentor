// Everything else under `/intake/`: a 404, never the SPA shell. The path is
// exempt from the site's sign-in (deploy/DEPLOY.md, "Bug-report intake"), and
// Access exempts every subpath with it, so nothing may answer here but the
// intake itself (bugreport.ts, which outranks this catch-all).
import type { Env, PagesFunction } from "../env";
import { handleIntakeMiss } from "../api/_bugreport";

export const onRequest: PagesFunction<Env> = () => handleIntakeMiss();

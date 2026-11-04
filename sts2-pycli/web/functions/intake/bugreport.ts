// `/intake/bugreport` — the signed-out write path; see ../api/_bugreport.ts.
import type { Env, PagesFunction } from "../env";
import { handleIntakeProbe, handlePost } from "../api/_bugreport";

export const onRequestGet: PagesFunction<Env> = () => handleIntakeProbe();
export const onRequestPost: PagesFunction<Env> = ({ request, env }) => handlePost(request, env, {}, "intake");

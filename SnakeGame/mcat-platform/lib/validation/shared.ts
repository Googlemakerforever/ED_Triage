import { z } from "zod";

export const sectionSchema = z.enum(["CP", "CARS", "BB", "PS"]);
export const idSchema = z.string().min(8);

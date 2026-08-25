export type ApiSuccess<T> = { ok: true; data: T };
export type ApiError = { ok: false; error: { code: string; message: string } };
export type ApiResponse<T> = ApiSuccess<T> | ApiError;

export type Pagination = {
  page: number;
  pageSize: number;
  total: number;
};

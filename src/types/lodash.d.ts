declare module 'lodash' {
  export type DebouncedFunc<T extends (...args: any[]) => any> = {
    (...args: Parameters<T>): ReturnType<T> | undefined;
    cancel(): void;
    flush(): ReturnType<T>;
    pending(): boolean;
  };

  export function debounce<T extends (...args: any[]) => any>(
    func: T,
    wait?: number,
    options?: { leading?: boolean; maxWait?: number; trailing?: boolean }
  ): DebouncedFunc<T>;

  export function throttle<T extends (...args: any[]) => any>(
    func: T,
    wait?: number,
    options?: { leading?: boolean; trailing?: boolean }
  ): T & { cancel(): void };

  export function clamp(value: number, lower?: number, upper?: number): number;
  export function get<TObject, TKey extends keyof TObject, TDefault = TObject[TKey]>(
    object: TObject | null | undefined,
    path: TKey | string | Array<string | number>,
    defaultValue?: TDefault
  ): TDefault;

  const lodash: {
    debounce: typeof debounce;
    throttle: typeof throttle;
    clamp: typeof clamp;
    get: typeof get;
  };

  export default lodash;
}

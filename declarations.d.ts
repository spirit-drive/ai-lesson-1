declare module '*.sass';
declare module '*.scss';
declare module '*.css';
declare module '*.jpg';
declare module '*.jpeg';
declare module '*.png';
declare module '*.svg';

declare module 'mnist' {
  function set(
    arg1: number,
    arg2: number
  ): {
    training: Array<{ input: number[]; output: number[] }>;
    test: Array<{ input: number[]; output: number[] }>;
  };
}

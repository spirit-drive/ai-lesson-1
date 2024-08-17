import React from 'react';
import clsx from 'clsx';
import s from './Canvas.module.scss';

export type CanvasProps = {
  className?: string;
  data: number[];
};

export const Canvas = ({ className, data }: CanvasProps) => {
  const size = Math.sqrt(data.length);
  if (size !== ~~size) throw new Error(`data must be a square`);

  return (
    <div className={clsx(s.root, className)}>
      {Array.from({ length: size }, (_, i) => (
        <div key={i} className={s.row}>
          {Array.from({ length: size }, (__, j) => (
            <div key={j} className={s.cell} style={{ opacity: data[i * size + j] }} />
          ))}
        </div>
      ))}
    </div>
  );
};

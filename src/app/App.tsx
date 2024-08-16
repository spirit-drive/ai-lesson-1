import React, { useState } from 'react';

import './App.css';
import { Canvas } from 'src/features/Canvas/Canvas';
import { nn, training } from 'src/features/core/NeuralNetwork/NeuraslNetwork';

function App() {
  const [number, setNumber] = useState(0);

  const back = () => setNumber((v) => (v === 0 ? training.length - 1 : v - 1));
  const next = () => setNumber((v) => (v === training.length - 1 ? 0 : v + 1));

  const data = training[number].input;
  const prediction = nn.forward(data).map((i) => Math.round(i * 100) / 100);

  const max = Math.max(...prediction);

  return (
    <div>
      <Canvas data={data} />
      <div>
        <button onClick={back} type="button">{`<`}</button>
        <button onClick={next} type="button">{`>`}</button>
      </div>
      <div>
        {prediction.map((item, i) => {
          return <div key={i} style={{ color: max === item ? 'red' : 'black' }}>{`${i}: ${item}`}</div>;
        })}
      </div>
    </div>
  );
}

export default App;

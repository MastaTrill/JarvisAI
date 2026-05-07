import React from "react";
import ReactDOM from "react-dom/client";

function App() {
  return (
    <div style={{ textAlign: "center", marginTop: 40 }}>
      <h1>Welcome to JarvisAI React App!</h1>
      <p>Your React environment is set up and running.</p>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);

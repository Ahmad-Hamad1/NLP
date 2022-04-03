import React, { useState } from "react";
import Header from "./components/header/Header";
import LoadingSpinner from "./components/LoadingSpinner/LoadingSpinner";
import classes from "./App.module.css";
import axios from "axios";

const models = [
  "Decision Tree",
  "Random Forest",
  "Naive Bayes",
  "Support Vector Machine",
  "KNN",
  "Artificial Neural network",
];

const modes = [
  "To train and test the model on the data set",
  "To run the model on a uer-input string",
];

const featureModels = [
  "TF-IDF with stemmer",
  "TF-IDF without stemmer",
  "Bag of Words with stemmer",
  "Bag of Words without stemmer",
];

const URL = "http://localhost:5000";

function App() {
  const [data, setData] = useState({ model: 1, mode: 1, featureModel: 1 });
  const [showResult, setShowResult] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState({});
  const [list, setList] = useState(0);
  const [str, setStr] = useState("");

  const listOne = [];
  const listTwo = [];
  const listThree = [];

  for (let i = 0; i < 6; i++) {
    listOne.push(
      <option key={i} value={i + 1}>
        {models[i]}
      </option>
    );
  }

  for (let i = 0; i < 2; i++) {
    listTwo.push(
      <option key={i + 6} value={i + 1}>
        {modes[i]}
      </option>
    );
  }

  for (let i = 0; i < 4; i++) {
    listThree.push(
      <option key={i + 8} value={i + 1}>
        {featureModels[i]}
      </option>
    );
  }

  const modelSelection = (evt) => {
    setData((prevData) => ({
      ...prevData,
      model: parseInt(evt.target.value),
    }));
  };

  const modeSelection = (evt) => {
    setData((prevData) => ({
      ...prevData,
      mode: parseInt(evt.target.value),
    }));
  };

  const featureSelection = (evt) => {
    setData((prevData) => ({
      ...prevData,
      featureModel: parseInt(evt.target.value),
    }));
  };

  const inputString = (evt) => {
    setStr(evt.target.value);
  };

  const nextClickListner = () => {
    setList((prevList) => {
      const mod = data.mode === 1 ? 2 : 3;
      prevList = ++prevList % mod;
      return prevList;
    });
  };

  const backClickListner = () => {
    setList((prevList) => {
      const mod = data.mode === 1 ? 2 : 3;
      prevList = ((--prevList % mod) + mod) % mod;
      return prevList;
    });
  };

  const requestData = async () => {
    setIsLoading(true);
    const res = await axios(URL, {
      method: "POST",
      data: {
        ...data,
        inputStr: str,
      },
    });
    setIsLoading(false);
    if (res !== 200) {
      //
    }

    setResult(res.data);
    setShowResult(true);
  };

  const tryAgain = () => {
    setList(0);
    setStr("");
    setShowResult(false);
    setData({ model: 1, mode: 1, featureModel: 1 });
  };

  const lists = [
    <div className={classes.inputs}>
      <label htmlFor="Models">Choose Model</label>
      <select id="Models" value={data.model} onChange={modelSelection}>
        {listOne}
      </select>
    </div>,
    <div className={classes.inputs}>
      <label htmlFor="Modes">Choose Mode</label>
      <select id="Modes" value={data.mode} onChange={modeSelection}>
        {listTwo}
      </select>
    </div>,
    <div className={classes.inputs}>
      <label htmlFor="Features">Choose Features</label>
      <select
        id="Features"
        value={data.featureModel}
        onChange={featureSelection}
      >
        {listThree}
      </select>
    </div>,
  ];

  return (
    <React.Fragment>
      <Header className={classes.header} />

      {!showResult && !isLoading && (
        <section className={classes["list"]}>
          <div className={classes.box}>
            {list === 0 && lists[0]}
            {list === 0 && lists[1]}
            {list === 0 && lists[2]}

            {list === 1 && data.mode === 2 && (
              <textarea
                onChange={inputString}
                value={str}
                className={classes.text}
                placeholder="أدخل النص باللغة العربية..."
              ></textarea>
            )}
            <div className={classes.buttons}>
              <button
                onClick={backClickListner}
                disabled={list === 0 ? true : false}
              >
                Back
              </button>
              {list === 0 && data.mode === 2 && (
                <button onClick={nextClickListner}>Next</button>
              )}
              {(data.mode === 1 || (list >= 1 && data.mode === 2)) && (
                <button
                  onClick={requestData}
                  disabled={
                    (list === 1 && data.mode === 1) ||
                    (list === 2 && data.mode === 2)
                      ? true
                      : false
                  }
                >
                  Result
                </button>
              )}
            </div>
          </div>
        </section>
      )}
      {showResult && !isLoading && (
        <section className={classes.list}>
          {data.mode === 2 && (
            <p className={classes.textAnswer}>
              The string <q>{str}</q> is{" "}
              <span
                className={`${
                  result.class === "Offensive"
                    ? classes.offensive
                    : classes.notOffensive
                }`}
              >
                {result.class}
              </span>
            </p>
          )}
          {data.mode === 1 && (
            <div>
              <p className={classes.modelName}>{models[data.model - 1]}</p>
              <p className={classes.feature}>
                Features extraction model:
              </p>
              <p className={`${classes.feature} ${classes.featureName}`}>
                {featureModels[data.featureModel - 1]}
              </p>
              <table className={classes.infoTable}>
                <caption>Confusion Matrix</caption>
                <thead>
                  <tr>
                    <th></th>
                    <th>Positive</th>
                    <th>Negative</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th>Positive</th>
                    <td>{result.tp}</td>
                    <td>{result.fp}</td>
                  </tr>
                  <tr>
                    <th>Negative</th>
                    <td>{result.fn}</td>
                    <td>{result.tn}</td>
                  </tr>
                </tbody>
              </table>
              <table className={classes.infoTable}>
                <caption>Measurements</caption>
                <thead>
                  <tr>
                    <th>Measure</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Precision</td>
                    <td>{result.Precision}%</td>
                  </tr>
                  <tr>
                    <td>Recall</td>
                    <td>{result.Recall}%</td>
                  </tr>
                  <tr>
                    <td>F1 - Score</td>
                    <td>{result["F1-score"]}%</td>
                  </tr>
                  <tr>
                    <td>Accuracy</td>
                    <td>{result.Accuracy}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
          <button onClick={tryAgain} className={classes.tryAgain}>
            Try Again
          </button>
        </section>
      )}
      {isLoading && <LoadingSpinner />}
    </React.Fragment>
  );
}

export default App;

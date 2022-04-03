import classes from "./Header.module.css";

const Header = (props) => {
  return (
    <header className={`${classes["main-header"]} ${props.className}`}>
      <p>Offensive Tweets Classification</p>
    </header>
  );
};

export default Header;

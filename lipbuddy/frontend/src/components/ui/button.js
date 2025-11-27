export const Button = ({ children, onClick, disabled }) => (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`px-4 py-2 bg-blue-500 text-white rounded-lg ${disabled ? "opacity-50" : ""}`}
    >
      {children}
    </button>
  );
  
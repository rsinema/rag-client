import './Typing.css';

const TypingBubble = () => (
    <div className="flex justify-start mb-2 w-full">
      <div className="bg-gray-600 text-white px-4 py-2 rounded-lg flex">
        <span className="typing-dot"></span>
        <span className="typing-dot"></span>
        <span className="typing-dot"></span>
      </div>
    </div>
  );

  export default TypingBubble;
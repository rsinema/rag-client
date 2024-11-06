import { useState } from "react";
import { FiArrowUp } from "react-icons/fi";

function App() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>(
    []
  );
  const [input, setInput] = useState("");
  const [currentUser, setCurrentUser] = useState("Brayden");

  const handleSendMessage = () => {
    if (input.trim() === "") return;

    setMessages([...messages, { role: "user", content: input }]);

    setTimeout(() => {
      setMessages((prevMessages) => [
        ...prevMessages,
        { role: "bot", content: "This is a response from ChatGPT." },
      ]);
    }, 1000);

    setInput("");
  };

  return (
    <>
      <div className="bg-zinc-800 h-screen">
        <div className="flex justify-center items-center text-white py-10">
          <h1 className="text-4xl">NLP RAG Project</h1>
          <div className="flex space-x-2">
            <button
              onClick={() => setCurrentUser("Brayden")}
              className={`px-4 py-2 rounded-full ${
                currentUser === "Brayden"
                  ? "bg-blue-500 text-white"
                  : "bg-gray-600 text-gray-300"
              }`}
            >
              Brayden
            </button>
            <button
              onClick={() => setCurrentUser("Riley")}
              className={`px-4 py-2 rounded-full ${
                currentUser === "Riley"
                  ? "bg-blue-500 text-white"
                  : "bg-gray-600 text-gray-300"
              }`}
            >
              Riley
            </button>
          </div>
        </div>

        {/* Display Current User */}
        <div className="text-center text-white text-xl mb-4">
          Current User: {currentUser}
        </div>

        {/* Message Display Area */}
        <div className="flex-grow overflow-y-auto px-4 py-2">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role === "user" ? "justify-end" : "justify-start"
              } mb-2`}
            >
              <div
                className={`max-w-xs px-4 py-2 rounded-lg ${
                  message.role === "user"
                    ? "bg-blue-500 text-white"
                    : "bg-gray-700 text-white"
                }`}
              >
                {message.content}
              </div>
            </div>
          ))}
        </div>

        {/* Footer with Input Bar */}
        <div className="absolute bottom-0 w-full h-24 text-white flex flex-col justify-center items-center my-2">
          <div className="flex w-[90%] px-4 py-2 bg-zinc-900 rounded-full items-center mx-4 my-2">
            <input
              type="text"
              placeholder="Message Chatbot"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
              className="flex-grow bg-transparent text-white border-none focus:outline-none px-2"
            />

            <button
              onClick={handleSendMessage}
              className="w-10 h-10 bg-gray-600 flex justify-center items-center rounded-full ml-2"
            >
              <FiArrowUp className="text-white" />
            </button>
          </div>
          <p>Created by Brayden Christensen and Riley Sinema</p>
        </div>
      </div>
    </>
  );
}

export default App;

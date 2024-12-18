import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:5116",
  headers: {
    "Content-Type": "application/json",
  },
});

export const triggerRAG = async (input: string, conversationId: string) => {
  try {
    const response = await api.post("/chat", {
      user_input: input,
      conversation_id: conversationId,
      timestamp: new Date().toISOString(),
    });
    return {
      bot_response: response.data.response,
      conversation_id: conversationId,
    };
  } catch (error) {
    console.error("Error in chat:", error);
    throw error;
  }
};

export const getBotResponse = async (input: any) => {
  try {
    const response = await api.get("/model-inference", {
      params: { user_input: input },
    });
    console.log("response:", response);
    return response.data;
  } catch (error) {
    console.error("Error getting bot response:", error);
    throw error;
  }
};

export default api;

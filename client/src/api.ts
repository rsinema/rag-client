import axios from 'axios';

const api = axios.create({
    baseURL: 'http://localhost:5001',
    headers: {
        'Content-Type': 'application/json',
    },
});

export const retrieveRelatedDocuments = async (input: any) => {
    try {
        const response = await api.get('/related-documents', {
            params: { user_input: input }
        });
        return response.data;
    } catch (error) {
        console.error('Error getting example data:', error);
        throw error;
    }
};

export const getBotResponse = async (input: any) => {
    try {
        const response = await api.get('/model-inference', {
            params: { user_input: input }
        });
        console.log('response:', response);
        return response.data;
    } catch (error) {
        console.error('Error getting bot response:', error);
        throw error;
    }
};

export default api;
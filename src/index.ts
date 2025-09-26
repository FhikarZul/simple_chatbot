import "dotenv/config";
import { z } from "zod";
import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
  Annotation,
} from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import readline from "readline";

type Contact = { name: string; phone: string; city: string };

let contacts: Contact[] = [
  { name: "Andi", phone: "08123456789", city: "Makassar" },
  { name: "Budi", phone: "08987654321", city: "Bone" },
];

// 1. ChatAnnotation dengan field intent
const ChatAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  intent: Annotation<
    "emotional" | "logical" | "general" | "contact_request" | undefined
  >(),
});

type ChatState = typeof ChatAnnotation.State;

// 2. Model LLM
const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
});

// 3. Classifier Node (pakai Zod structured output)
async function classifierNode(state: ChatState): Promise<ChatState> {
  const last = state.messages[state.messages.length - 1];

  const schema = z.object({
    intent: z.enum(["emotional", "logical", "general", "contact_request"]),
  });

  const resp = await model.withStructuredOutput(schema).invoke([
    new SystemMessage(
      `Classify the user message as either:
       - 'emotional': for emotional support or feelings
       - 'logical': for facts, information, or general queries not covered by the above
       - 'general': for greetings (e.g., "hello", "hi") or simple small talk
       - 'contact_request': if the user asks for contact information, phone number, contact list
       `
    ),
    new HumanMessage(`Message: "${last.content}"`),
  ]);

  return { ...state, intent: resp.intent };
}

async function contactNode(state: ChatState): Promise<ChatState> {
  const last = state.messages[state.messages.length - 1];

  const resp = await model.invoke([
    new SystemMessage(`Kamu adalah asisten kontak. 
  Gunakan data kontak yang diberikan untuk menjawab pertanyaan user.
  Jika user bertanya nomor seseorang, cari di daftar. 
  Jika tidak ditemukan, jawab dengan sopan "kontak tidak ditemukan". 
  Jika user minta menambahkan kontak, katakan bahwa fungsi tambah belum tersedia 
  (kecuali kalau nanti ditambahkan update DB).
      
  Daftar kontak:
  ${JSON.stringify(contacts, null, 2)}
      `),
    new HumanMessage(`Pesan user: "${last.content}"`),
  ]);

  return { ...state, messages: [...state.messages, resp] };
}

// 4. Logical Node
async function logicalNode(state: ChatState): Promise<ChatState> {
  const last = state.messages[state.messages.length - 1];
  const resp = await model.invoke([
    new SystemMessage(
      `You are a purely logical assistant. Focus only on facts and information.
       Provide clear, concise answers based on logic and evidence.
       Do not address emotions or provide emotional support.
       Be direct and straightforward in your responses.`
    ),
    new HumanMessage(`Message: "${last.content}"`),
  ]);

  return { ...state, messages: [...state.messages, resp] };
}

// 5. Emotional Node
async function emotionalNode(state: ChatState): Promise<ChatState> {
  const last = state.messages[state.messages.length - 1];
  const resp = await model.invoke([
    new SystemMessage(
      `You are a compassionate therapist. Focus on the emotional aspects of the user's message.
       Show empathy, validate their feelings, and help them process their emotions.
       Ask thoughtful questions to help them explore their feelings more deeply.
       Avoid giving logical solutions unless explicitly asked.`
    ),
    new HumanMessage(`Message: "${last.content}"`),
  ]);

  return { ...state, messages: [...state.messages, resp] };
}

// 6. General Node
async function generalNode(state: ChatState): Promise<ChatState> {
  const last = state.messages[state.messages.length - 1];
  const resp = await model.invoke([
    new SystemMessage(
      `You are a friendly chatbot. Handle greetings and small talk casually.
       Be warm, polite, and concise.`
    ),
    new HumanMessage(`Message: "${last.content}"`),
  ]);

  return { ...state, messages: [...state.messages, resp] };
}

// 7. Build Graph
const workflow = new StateGraph(ChatAnnotation)
  .addNode("classifier", classifierNode)
  .addNode("logical", logicalNode)
  .addNode("emotional", emotionalNode)
  .addNode("general", generalNode)
  .addNode("contact_request", contactNode)

  .addEdge(START, "classifier")
  .addConditionalEdges("classifier", (state) => state.intent ?? "logical", {
    logical: "logical",
    emotional: "emotional",
    general: "general",
    contact_request: "contact_request",
  })

  .addEdge("logical", END)
  .addEdge("emotional", END)
  .addEdge("general", END)
  .addEdge("contact_request", END);

const app = workflow.compile();

// 8. CLI Chat Loop
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

async function main() {
  let state: ChatState = { messages: [], intent: undefined };

  while (true) {
    const userInput: string = await new Promise((resolve) =>
      rl.question("👤 Kamu: ", resolve)
    );

    if (userInput.toLowerCase() === "exit") {
      console.log("👋 Chat selesai.");
      break;
    }

    // masukkan pesan user
    state.messages.push(new HumanMessage(userInput));

    // panggil graph
    state = await app.invoke(state);

    // ambil respons terakhir
    const last = state.messages[state.messages.length - 1];
    console.log("🤖 Bot:", last.content);
  }

  rl.close();
}

main();

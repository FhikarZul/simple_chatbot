// Import konfigurasi environment variables dari file .env
import "dotenv/config";
// Import Zod untuk validasi schema dan structured output
import { z } from "zod";
// Import komponen utama LangGraph untuk membangun state graph
import {
  StateGraph,        // Class untuk membuat graph workflow
  MessagesAnnotation, // Annotation untuk mengelola pesan dalam state
  START,             // Node awal dari graph
  END,               // Node akhir dari graph
  Annotation,        // Class untuk membuat custom annotation
} from "@langchain/langgraph";
// Import model LLM OpenAI
import { ChatOpenAI } from "@langchain/openai";
// Import tipe pesan untuk LangChain
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
// Import readline untuk interface command line
import readline from "readline";

// Tipe data untuk menyimpan informasi kontak
type Contact = { name: string; phone: string; city: string };

// Database kontak sederhana (dalam implementasi nyata, ini akan disimpan di database)
let contacts: Contact[] = [
  { name: "Andi", phone: "08123456789", city: "Makassar" },
  { name: "Budi", phone: "08987654321", city: "Bone" },
];

// 1. Definisi State Annotation untuk LangGraph
// ChatAnnotation menggabungkan MessagesAnnotation dengan field intent tambahan
const ChatAnnotation = Annotation.Root({
  // Menggunakan MessagesAnnotation untuk mengelola riwayat pesan
  ...MessagesAnnotation.spec,
  // Menambahkan field intent untuk menyimpan hasil klasifikasi
  intent: Annotation<
    "emotional" | "logical" | "general" | "contact_request" | undefined
  >(),
});

// Tipe state yang akan digunakan di seluruh graph
type ChatState = typeof ChatAnnotation.State;

// 2. Konfigurasi Model LLM OpenAI
const model = new ChatOpenAI({
  model: "gpt-4o",    // Menggunakan GPT-4o untuk performa terbaik
  temperature: 0,     // Temperature 0 untuk konsistensi respons
});

// 3. Node untuk Klasifikasi Intent (menggunakan Zod structured output)
// Node ini akan menganalisis pesan terakhir dan menentukan intent pengguna
async function classifierNode(state: ChatState): Promise<ChatState> {
  // Ambil pesan terakhir dari state
  const last = state.messages[state.messages.length - 1];

  // Definisi schema untuk structured output menggunakan Zod
  const schema = z.object({
    intent: z.enum(["emotional", "logical", "general", "contact_request"]),
  });

  // Panggil model dengan structured output untuk mendapatkan klasifikasi yang konsisten
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

  // Update state dengan intent yang telah diklasifikasi
  return { ...state, intent: resp.intent };
}

// 4. Node untuk Menangani Permintaan Kontak
// Node ini akan dipanggil ketika intent adalah "contact_request"
async function contactNode(state: ChatState): Promise<ChatState> {
  // Ambil pesan terakhir dari state
  const last = state.messages[state.messages.length - 1];

  // Panggil model dengan konteks sebagai asisten kontak
  const resp = await model.invoke([
    new SystemMessage(`You are a contact assistant.
    Use the contact information provided to answer user questions.
    If a user asks for someone's number, look it up in the list.
    If you can't find it, politely reply "contact not found."
    If a user asks to add a contact, explain that the add function isn't available yet.
    (Unless it's added later in a database update).
      
  Daftar kontak:
  ${JSON.stringify(contacts, null, 2)}
      `),
    new HumanMessage(`Pesan user: "${last.content}"`),
  ]);

  // Tambahkan respons model ke dalam messages dan return state yang telah diupdate
  return { ...state, messages: [...state.messages, resp] };
}

// 5. Node untuk Menangani Pertanyaan Logis
// Node ini akan dipanggil ketika intent adalah "logical"
async function logicalNode(state: ChatState): Promise<ChatState> {
  // Ambil pesan terakhir dari state
  const last = state.messages[state.messages.length - 1];
  
  // Panggil model dengan persona asisten logis
  const resp = await model.invoke([
    new SystemMessage(
      `You are a purely logical assistant. Focus only on facts and information.
       Provide clear, concise answers based on logic and evidence.
       Do not address emotions or provide emotional support.
       Be direct and straightforward in your responses.`
    ),
    new HumanMessage(`Message: "${last.content}"`),
  ]);

  // Tambahkan respons model ke dalam messages dan return state yang telah diupdate
  return { ...state, messages: [...state.messages, resp] };
}

// 6. Node untuk Menangani Dukungan Emosional
// Node ini akan dipanggil ketika intent adalah "emotional"
async function emotionalNode(state: ChatState): Promise<ChatState> {
  // Ambil pesan terakhir dari state
  const last = state.messages[state.messages.length - 1];
  
  // Panggil model dengan persona terapis yang berempati
  const resp = await model.invoke([
    new SystemMessage(
      `You are a compassionate therapist. Focus on the emotional aspects of the user's message.
       Show empathy, validate their feelings, and help them process their emotions.
       Ask thoughtful questions to help them explore their feelings more deeply.
       Avoid giving logical solutions unless explicitly asked.`
    ),
    new HumanMessage(`Message: "${last.content}"`),
  ]);

  // Tambahkan respons model ke dalam messages dan return state yang telah diupdate
  return { ...state, messages: [...state.messages, resp] };
}

// 7. Node untuk Menangani Salam dan Obrolan Ringan
// Node ini akan dipanggil ketika intent adalah "general"
async function generalNode(state: ChatState): Promise<ChatState> {
  // Ambil pesan terakhir dari state
  const last = state.messages[state.messages.length - 1];
  
  // Panggil model dengan persona chatbot yang ramah
  const resp = await model.invoke([
    new SystemMessage(
      `You are a friendly chatbot. Handle greetings and small talk casually.
       Be warm, polite, and concise.`
    ),
    new HumanMessage(`Message: "${last.content}"`),
  ]);

  // Tambahkan respons model ke dalam messages dan return state yang telah diupdate
  return { ...state, messages: [...state.messages, resp] };
}

// 8. Membangun State Graph dengan LangGraph
// Graph ini akan mengatur alur eksekusi berdasarkan hasil klasifikasi intent
const workflow = new StateGraph(ChatAnnotation)
  // Menambahkan semua nodes ke dalam graph
  .addNode("classifier", classifierNode)        // Node untuk klasifikasi intent
  .addNode("logical", logicalNode)              // Node untuk pertanyaan logis
  .addNode("emotional", emotionalNode)          // Node untuk dukungan emosional
  .addNode("general", generalNode)              // Node untuk salam dan obrolan ringan
  .addNode("contact_request", contactNode)      // Node untuk permintaan kontak

  // Menambahkan edges untuk mengatur alur eksekusi
  .addEdge(START, "classifier")                 // Mulai dari node classifier
  .addConditionalEdges("classifier", (state) => state.intent ?? "logical", {
    // Conditional edges berdasarkan hasil klasifikasi intent
    logical: "logical",           // Jika intent = "logical" → ke logical node
    emotional: "emotional",       // Jika intent = "emotional" → ke emotional node
    general: "general",           // Jika intent = "general" → ke general node
    contact_request: "contact_request", // Jika intent = "contact_request" → ke contact node
  })

  // Semua nodes akan berakhir di END
  .addEdge("logical", END)
  .addEdge("emotional", END)
  .addEdge("general", END)
  .addEdge("contact_request", END);

// Compile graph menjadi executable application
const app = workflow.compile();

// 9. Setup Command Line Interface (CLI)
// Membuat interface untuk interaksi dengan user melalui terminal
const rl = readline.createInterface({
  input: process.stdin,   // Input dari keyboard
  output: process.stdout, // Output ke terminal
});

// 10. Fungsi Utama untuk Chat Loop
async function main() {
  // Inisialisasi state awal dengan messages kosong dan intent undefined
  let state: ChatState = { messages: [], intent: undefined };

  // Loop utama untuk chat interaktif
  while (true) {
    // Menunggu input dari user
    const userInput: string = await new Promise((resolve) =>
      rl.question("👤 Kamu: ", resolve)
    );

    // Cek apakah user ingin keluar
    if (userInput.toLowerCase() === "exit") {
      console.log("👋 Chat selesai.");
      break;
    }

    // Tambahkan pesan user ke dalam state messages
    state.messages.push(new HumanMessage(userInput));

    // Jalankan graph dengan state yang telah diupdate
    // LangGraph akan memproses pesan melalui semua nodes sesuai dengan alur yang telah didefinisikan
    state = await app.invoke(state);

    // Ambil respons terakhir dari bot
    const last = state.messages[state.messages.length - 1];
    console.log("🤖 Bot:", last.content);
  }

  // Tutup readline interface setelah loop selesai
  rl.close();
}

// Jalankan fungsi main
main();

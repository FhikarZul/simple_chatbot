# Simple Chatbot with LangGraph

Sebuah chatbot sederhana yang dibangun menggunakan LangGraph dengan kemampuan untuk mengklasifikasikan intent dan memberikan respons yang sesuai berdasarkan tipe pesan pengguna.

## 🚀 Fitur

- **Intent Classification**: Mengklasifikasikan pesan pengguna menjadi 4 kategori:

  - `emotional`: Dukungan emosional atau perasaan
  - `logical`: Fakta, informasi, atau pertanyaan umum
  - `general`: Salam dan obrolan ringan
  - `contact_request`: Permintaan informasi kontak

- **Multi-Modal Responses**: Setiap intent memiliki node khusus yang memberikan respons yang disesuaikan
- **Contact Management**: Dapat mencari dan menampilkan informasi kontak yang tersimpan
- **Interactive CLI**: Interface command line yang user-friendly

## 🛠️ Teknologi yang Digunakan

- **LangGraph**: Framework untuk membangun aplikasi AI stateful
- **LangChain**: Library untuk integrasi dengan LLM
- **OpenAI GPT-4o**: Model bahasa untuk pemrosesan dan klasifikasi
- **TypeScript**: Bahasa pemrograman utama
- **Zod**: Validasi schema dan structured output

## 📋 Prasyarat

- Node.js (versi 16 atau lebih baru)
- npm atau yarn
- OpenAI API key

## ⚙️ Instalasi

1. **Clone repository**

   ```bash
   git clone <repository-url>
   cd simple_chatbot
   ```

2. **Install dependencies**

   ```bash
   npm install
   ```

3. **Setup environment variables**
   Buat file `.env` di root directory dan tambahkan:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🏃‍♂️ Menjalankan Aplikasi

### Development Mode

```bash
npm start
```

### Build dan Production

```bash
# Build TypeScript ke JavaScript
npm run build

# Jalankan versi yang sudah di-build
npm run serve
```

## 🎯 Cara Penggunaan

1. Jalankan aplikasi dengan `npm start`
2. Ketik pesan Anda di prompt "👤 Kamu: "
3. Bot akan mengklasifikasikan intent dan memberikan respons yang sesuai
4. Ketik `exit` untuk keluar dari aplikasi

### Contoh Interaksi

```
👤 Kamu: Halo
🤖 Bot: Hello! How can I help you today?

👤 Kamu: Saya sedih hari ini
🤖 Bot: I'm sorry to hear that you're feeling sad today. It's completely normal to have difficult days...

👤 Kamu: Berapa 2 + 2?
🤖 Bot: 2 + 2 equals 4.

👤 Kamu: Apa nomor telepon Andi?
🤖 Bot: Nomor telepon Andi adalah 08123456789.
```

## 🏗️ Arsitektur

Aplikasi menggunakan LangGraph StateGraph dengan struktur berikut:

```
START → classifier → [logical | emotional | general | contact_request] → END
```

### Nodes:

- **classifier**: Mengklasifikasikan intent pesan menggunakan structured output
- **logical**: Memberikan respons faktual dan logis
- **emotional**: Memberikan dukungan emosional dan empati
- **general**: Menangani salam dan obrolan ringan
- **contact_request**: Mencari dan menampilkan informasi kontak

### State:

```typescript
type ChatState = {
  messages: Message[];
  intent: "emotional" | "logical" | "general" | "contact_request" | undefined;
};
```

## 📁 Struktur Project

```
simple_chatbot/
├── src/
│   └── index.ts          # File utama aplikasi
├── package.json          # Dependencies dan scripts
├── tsconfig.json         # Konfigurasi TypeScript
└── README.md            # Dokumentasi
```

## 🔧 Konfigurasi

### Environment Variables

| Variable         | Deskripsi           | Wajib |
| ---------------- | ------------------- | ----- |
| `OPENAI_API_KEY` | API key dari OpenAI | ✅    |

### Model Configuration

Aplikasi menggunakan GPT-4o dengan konfigurasi:

- Temperature: 0 (untuk konsistensi)
- Model: gpt-4o

## 🚧 Pengembangan

### Menambahkan Intent Baru

1. Tambahkan intent baru di enum schema Zod (line 42)
2. Buat node function baru
3. Tambahkan node ke graph
4. Update conditional edges

### Menambahkan Fitur Kontak

Saat ini aplikasi hanya dapat mencari kontak. Untuk menambahkan fitur create/update/delete:

1. Implementasikan database atau file storage
2. Buat nodes untuk CRUD operations
3. Update classifier untuk mengenali operasi CRUD
4. Tambahkan validation untuk data kontak

## 🤝 Contributing

1. Fork repository
2. Buat feature branch (`git checkout -b feature/amazing-feature`)
3. Commit perubahan (`git commit -m 'Add amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Buat Pull Request

## 📄 License

ISC License - lihat file [LICENSE](LICENSE) untuk detail.

## 🐛 Troubleshooting

### Error: OpenAI API Key not found

- Pastikan file `.env` ada di root directory
- Pastikan `OPENAI_API_KEY` sudah di-set dengan benar

### Error: Cannot find module

- Jalankan `npm install` untuk menginstall dependencies
- Pastikan Node.js versi 16 atau lebih baru

### Bot tidak merespons dengan benar

- Periksa koneksi internet
- Pastikan API key valid dan memiliki quota yang cukup
- Cek logs untuk error messages

## 📞 Support

Jika mengalami masalah atau memiliki pertanyaan, silakan buat issue di repository ini.

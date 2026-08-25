import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class ChatServer {
    private static final int PORT = 3000;
    private static final File PUBLIC_DIR = new File("chatgpt-like/public");

    public static void main(String[] args) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/api/chat", new ChatHandler());
        server.createContext("/", new StaticHandler());
        server.setExecutor(null);
        server.start();
        System.out.println("Chat app running at http://localhost:" + PORT);
    }

    static class ChatHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod())) {
                sendText(exchange, 405, "Method Not Allowed", "text/plain");
                return;
            }

            String apiKey = System.getenv("OPENAI_API_KEY");
            if (apiKey == null || apiKey.trim().isEmpty()) {
                sendText(exchange, 500, "{\"error\":\"Missing OPENAI_API_KEY environment variable.\"}", "application/json");
                return;
            }

            String requestBody = new String(readAll(exchange.getRequestBody()), StandardCharsets.UTF_8);

            HttpURLConnection conn = null;
            try {
                URL url = new URL("https://api.openai.com/v1/chat/completions");
                conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setRequestProperty("Authorization", "Bearer " + apiKey);
                conn.setDoOutput(true);

                OutputStream os = conn.getOutputStream();
                os.write(requestBody.getBytes(StandardCharsets.UTF_8));
                os.flush();
                os.close();

                int status = conn.getResponseCode();
                InputStream responseStream = status >= 400 ? conn.getErrorStream() : conn.getInputStream();
                String responseBody = new String(readAll(responseStream), StandardCharsets.UTF_8);
                sendText(exchange, status, responseBody, "application/json");
            } catch (IOException e) {
                sendText(exchange, 500, "{\"error\":\"Server error: " + escapeForJson(e.getMessage()) + "\"}", "application/json");
            } finally {
                if (conn != null) conn.disconnect();
            }
        }
    }

    static class StaticHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) {
                sendText(exchange, 405, "Method Not Allowed", "text/plain");
                return;
            }

            String path = exchange.getRequestURI().getPath();
            if (path.equals("/")) {
                path = "/index.html";
            }

            File file = new File(PUBLIC_DIR, path.substring(1));
            String canonicalPublic = PUBLIC_DIR.getCanonicalPath();
            String canonicalFile = file.getCanonicalPath();

            if (!canonicalFile.startsWith(canonicalPublic) || !file.exists() || file.isDirectory()) {
                sendText(exchange, 404, "Not found", "text/plain");
                return;
            }

            byte[] data = readFile(file);
            Headers headers = exchange.getResponseHeaders();
            headers.set("Content-Type", contentType(file.getName()));
            exchange.sendResponseHeaders(200, data.length);
            OutputStream os = exchange.getResponseBody();
            os.write(data);
            os.close();
        }
    }

    private static void sendText(HttpExchange exchange, int status, String body, String contentType) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", contentType + "; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        OutputStream os = exchange.getResponseBody();
        os.write(bytes);
        os.close();
    }

    private static byte[] readAll(InputStream in) throws IOException {
        if (in == null) return new byte[0];
        try (BufferedInputStream bis = new BufferedInputStream(in);
             ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = bis.read(buffer)) != -1) {
                baos.write(buffer, 0, read);
            }
            return baos.toByteArray();
        }
    }

    private static byte[] readFile(File file) throws IOException {
        try (FileInputStream fis = new FileInputStream(file);
             ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = fis.read(buffer)) != -1) {
                baos.write(buffer, 0, read);
            }
            return baos.toByteArray();
        }
    }

    private static String contentType(String name) {
        if (name.endsWith(".html")) return "text/html";
        if (name.endsWith(".css")) return "text/css";
        if (name.endsWith(".js")) return "application/javascript";
        return "application/octet-stream";
    }

    private static String escapeForJson(String value) {
        if (value == null) return "";
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}

package utilities;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.SourceDataLine;
import javax.sound.sampled.UnsupportedAudioFileException;

public class Sound {
    private static final String LOVE_STORY_FILE = "Love_Story.wav";
    private static final String DEFAULT_BG_FILE = "bg.wav";
    private Clip clip;
    private static Sound bgMusicInstance;
    private static Thread bgFallbackThread;
    private static volatile boolean bgFallbackRunning = false;

    public Sound(String filename) {
        try {
            InputStream resource = getClass().getResourceAsStream("/resources/sounds/" + filename);
            if (resource == null) {
                resource = getClass().getResourceAsStream("/sounds/" + filename);
            }
            if (resource == null) {
                try {
                    resource = new FileInputStream("resources/sounds/" + filename);
                } catch (IOException ignored) {
                    resource = null;
                }
            }
            if (resource == null) {
                try {
                    resource = new FileInputStream("sounds/" + filename);
                } catch (IOException ignored) {
                    resource = null;
                }
            }
            if (resource == null) {
                return;
            }

            BufferedInputStream bufferedStream = new BufferedInputStream(resource);
            AudioInputStream audioStream = AudioSystem.getAudioInputStream(bufferedStream);
            clip = AudioSystem.getClip();
            clip.open(audioStream);
        } catch (IOException | UnsupportedAudioFileException | LineUnavailableException e) {
            clip = null;
        }
    }

    public void play() {
        if (clip != null) {
            clip.setFramePosition(0);
            clip.start();
        }
    }

    public void loop() {
        if (clip != null) {
            clip.loop(Clip.LOOP_CONTINUOUSLY);
            clip.start();
        }
    }

    public void stop() {
        if (clip != null && clip.isRunning()) {
            clip.stop();
        }
    }

    public static void playBackgroundMusic() {
        stopFallbackBackground();
        if (bgMusicInstance == null) {
            bgMusicInstance = new Sound(LOVE_STORY_FILE);
            if (bgMusicInstance.clip == null) {
                bgMusicInstance = new Sound(DEFAULT_BG_FILE);
            }
        }

        if (bgMusicInstance.clip != null) {
            bgMusicInstance.clip.stop();
            bgMusicInstance.clip.setFramePosition(0);
            bgMusicInstance.loop();
        } else {
            startFallbackBackground();
        }
    }

    public static void stopBackgroundMusic() {
        stopFallbackBackground();
        if (bgMusicInstance != null && bgMusicInstance.clip != null) {
            bgMusicInstance.clip.stop();
            bgMusicInstance.clip.close();
        }
        bgMusicInstance = null;
    }

    public static void playEatSound() {
        Sound eat = new Sound("eat.wav");
        if (eat.clip != null) {
            eat.play();
            return;
        }
        playEatFallback();
    }

    public static void playGameOverSound() {
        Sound gameOver = new Sound("gameover.wav");
        if (gameOver.clip != null) {
            gameOver.play();
            return;
        }
        playGameOverFallback();
    }

    private static void startFallbackBackground() {
        if (bgFallbackRunning) {
            return;
        }

        bgFallbackRunning = true;
        bgFallbackThread = new Thread(() -> {
            int[] melody = {392, 440, 523, 440, 392, 330, 349, 392};
            int[] lengths = {180, 180, 220, 180, 180, 180, 180, 220};

            while (bgFallbackRunning) {
                for (int i = 0; i < melody.length && bgFallbackRunning; i++) {
                    playTone(melody[i], lengths[i], 0.10);
                    try {
                        Thread.sleep(30);
                    } catch (InterruptedException ignored) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                }
            }
        }, "snake-bg-fallback");
        bgFallbackThread.setDaemon(true);
        bgFallbackThread.start();
    }

    private static void stopFallbackBackground() {
        bgFallbackRunning = false;
        if (bgFallbackThread != null) {
            bgFallbackThread.interrupt();
            bgFallbackThread = null;
        }
    }

    private static void playEatFallback() {
        Thread eatThread = new Thread(() -> {
            playTone(880, 70, 0.20);
            playTone(1175, 60, 0.20);
        }, "snake-eat-fallback");
        eatThread.setDaemon(true);
        eatThread.start();
    }

    private static void playGameOverFallback() {
        Thread gameOverThread = new Thread(() -> {
            playTone(440, 140, 0.20);
            playTone(330, 160, 0.20);
            playTone(220, 220, 0.20);
        }, "snake-gameover-fallback");
        gameOverThread.setDaemon(true);
        gameOverThread.start();
    }

    private static void playTone(int hz, int durationMs, double volume) {
        float sampleRate = 44100f;
        AudioFormat format = new AudioFormat(sampleRate, 16, 1, true, false);
        int sampleCount = (int) (durationMs * sampleRate / 1000.0);
        byte[] output = new byte[sampleCount * 2];
        double angular = 2.0 * Math.PI * hz / sampleRate;
        double safeVolume = Math.max(0.0, Math.min(1.0, volume));

        for (int i = 0; i < sampleCount; i++) {
            short sample = (short) (Math.sin(i * angular) * Short.MAX_VALUE * safeVolume);
            output[2 * i] = (byte) (sample & 0xff);
            output[2 * i + 1] = (byte) ((sample >> 8) & 0xff);
        }

        try {
            DataLine.Info info = new DataLine.Info(SourceDataLine.class, format);
            SourceDataLine line = (SourceDataLine) AudioSystem.getLine(info);
            line.open(format);
            line.start();
            line.write(output, 0, output.length);
            line.drain();
            line.stop();
            line.close();
        } catch (LineUnavailableException ignored) {
            // Audio device unavailable; fail silently.
        }
    }
}

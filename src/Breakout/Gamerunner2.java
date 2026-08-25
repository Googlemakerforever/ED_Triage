package Breakout;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Toolkit;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.UnsupportedAudioFileException;

import game.GDV5;

public class Gamerunner2 extends GDV5 {

    private Paddle paddle;
    private Ball ball;
    private ArrayList<Brick> bricks;
    private ArrayList<PowerUp> powerUps;

    private int lives = 10;
    private boolean endScreenShown = false;

    private Clip bgMusicClip;
    private long lastPaddleSoundNanos = 0;
    private static final long PADDLE_SOUND_COOLDOWN_NANOS = 90_000_000L;

    public Gamerunner2() {
        paddle = new Paddle(250, 500, 100, 20);
        ball = new Ball(390, 300, 20);

        bricks = new ArrayList<Brick>();
        powerUps = new ArrayList<PowerUp>();

        setupBricks();

        Thread bgLoader = new Thread(this::startBackgroundMusic, "breakout-bg-loader");
        bgLoader.setDaemon(true);
        bgLoader.start();
    }

    @Override
    public void update() {
        if (endScreenShown) {
            return;
        }

        handlePaddleMovement();
        ball.move();
        handleWallBounce();
        handleBottomHit();
        handlePaddleCollision();

        for (Brick b : bricks) {
            b.update();
        }

        handleBrickCollisions();
        handlePowerUps();
    }

    private void handlePaddleMovement() {
        if (KeysPressed[37]) {
            paddle.moveLeft();
        }
        if (KeysPressed[39]) {
            paddle.moveRight(getMaxWindowX());
        }
    }

    private void handleWallBounce() {
        if (ball.getXPos() <= 0 || ball.getXPos() + ball.getDiameter() >= getMaxWindowX()) {
            ball.bounceX();
        }

        if (ball.getYPos() <= 0) {
            ball.bounceY();
        }
    }

    private void handleBottomHit() {
        if (ball.getYPos() + ball.getDiameter() >= getMaxWindowY()) {
            lives--;

            if (lives <= 0 && !endScreenShown) {
                endScreenShown = true;
                stopBackgroundMusic();

                EndScreen end = new EndScreen();
                end.start();

                this.dispose();
                return;
            }

            resetBall();
        }
    }

    private void resetBall() {
        ball.setXPos(getMaxWindowX() / 2 - ball.getDiameter() / 2);
        ball.setYPos(getMaxWindowY() / 2);

        if (ball.getDy() > 0) {
            ball.bounceY();
        }
    }

    private void handlePaddleCollision() {
        if (ball.getBounds().intersects(paddle.getBounds())) {
            ball.setYPos(paddle.getY() - ball.getDiameter());
            ball.bounceY();
            playPaddleBounceSound();
        }
    }

    private void handleBrickCollisions() {
        for (Brick b : bricks) {
            if (b.isVisible() && ball.getBounds().intersects(b.getBounds())) {
                int dir = GDV5.collisionDirection(
                        b.getBounds(),
                        ball.getBounds(),
                        ball.getDx(),
                        ball.getDy());

                if (dir == 0 || dir == 2) {
                    ball.bounceX();
                } else {
                    ball.bounceY();
                }

                b.hit();
                spawnPowerUp(b);
                break;
            }
        }
    }

    private void spawnPowerUp(Brick b) {
        int pillWidth = 6;
        int pillHeight = 20;

        int x = b.getX() + b.getWidth() / 2 - pillWidth / 2;
        int y = b.getY() + b.getHeight();
        int roll = ThreadLocalRandom.current().nextInt(3);
        PowerUp.Type type;

        if (roll == 0) {
            type = PowerUp.Type.RED;
        } else if (roll == 1) {
            type = PowerUp.Type.YELLOW;
        } else {
            type = PowerUp.Type.GREEN;
        }

        powerUps.add(new PowerUp(x, y, pillWidth, pillHeight, type));
    }

    private void handlePowerUps() {
        for (PowerUp p : powerUps) {
            if (!p.isActive()) {
                continue;
            }

            p.move();

            if (p.getY() > getMaxWindowY()) {
                p.setActive(false);
                continue;
            }

            if (p.getBounds().intersects(paddle.getBounds())) {
                if (p.getType() == PowerUp.Type.RED) {
                    ball.slowDown();
                } else if (p.getType() == PowerUp.Type.GREEN) {
                    paddle.increaseWidth(20, getMaxWindowX());
                } else {
                    paddle.increaseSpeed(3);
                }
                p.setActive(false);
            }
        }

        powerUps.removeIf(p -> !p.isActive());
    }

    @Override
    public void draw(Graphics2D win) {
        win.setColor(Color.BLACK);
        win.fillRect(0, 0, getMaxWindowX(), getMaxWindowY());

        paddle.draw(win);
        ball.draw(win);

        for (Brick b : bricks) {
            b.draw(win);
        }
        for (PowerUp p : powerUps) {
            p.draw(win);
        }

        win.setColor(Color.WHITE);
        win.drawString("Lives: " + lives, getMaxWindowX() - 100, getMaxWindowY() - 20);
    }

    private void setupBricks() {
        bricks.clear();

        int rows = 4;
        int cols = 8;
        int brickWidth = 80;
        int brickHeight = 25;
        int startX = 50;
        int startY = 50;
        int gap = 5;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (c % 2 != 0) {
                    continue;
                }

                int x = startX + c * (brickWidth + gap);
                int y = startY + r * (brickHeight + gap);

                bricks.add(new Brick(x, y, brickWidth, brickHeight));
            }
        }
    }

    private void startBackgroundMusic() {
        String[] candidates = {
                "resources/sounds/Love_Story.wav",
                "../SnakeGame/resources/sounds/Love_Story.wav",
                "SnakeGame/resources/sounds/Love_Story.wav",
                "/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/resources/sounds/Love_Story.wav"
        };

        for (String candidate : candidates) {
            File file = new File(candidate);
            if (!file.exists()) {
                continue;
            }

            try {
                AudioInputStream audioStream = AudioSystem.getAudioInputStream(file);
                bgMusicClip = AudioSystem.getClip();
                bgMusicClip.open(audioStream);
                bgMusicClip.loop(Clip.LOOP_CONTINUOUSLY);
                bgMusicClip.start();
                return;
            } catch (UnsupportedAudioFileException | IOException | LineUnavailableException ignored) {
                bgMusicClip = null;
            }
        }
    }

    private void stopBackgroundMusic() {
        if (bgMusicClip != null) {
            bgMusicClip.stop();
            bgMusicClip.close();
            bgMusicClip = null;
        }
    }

    private void playPaddleBounceSound() {
        long now = System.nanoTime();
        if (now - lastPaddleSoundNanos < PADDLE_SOUND_COOLDOWN_NANOS) {
            return;
        }
        lastPaddleSoundNanos = now;
        Toolkit.getDefaultToolkit().beep();
    }

    public static void main(String[] args) {
        SplashScreen splash = new SplashScreen();
        splash.start();
    }
}

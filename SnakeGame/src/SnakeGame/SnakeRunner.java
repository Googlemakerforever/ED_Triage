package SnakeGame;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.event.KeyEvent;

import utilities.GDV5;
import utilities.Sound;

public class SnakeRunner extends GDV5 {
    private Board board;
    private Snake snake;
    private powerUps powerUp;
    private int count;
    private int state; // 0: intro, 1: instructions, 2: gameplay, 3: game over
    public static boolean snakeDead;
    private int movement;
    private final int dMovement = 4;

    public SnakeRunner() {
        oriGame();
    }

    public static void main(String[] args) {
        SnakeRunner s = new SnakeRunner();
        s.start();
    }

    @Override
    public void update() {
        if (state == 0) {
            handleIntroPage();
        } else if (state == 1) {
            handleInstructionsPage();
        } else if (state == 2) {
            handleGameplay();
        } else if (state == 3) {
            handleGameOver();
        }
    }

    private void handleIntroPage() {
        if (GDV5.KeysPressed[KeyEvent.VK_1]) {
            GDV5.KeysPressed[KeyEvent.VK_1] = false;
            movement = 11;
            state = 1;
        } else if (GDV5.KeysPressed[KeyEvent.VK_2]) {
            GDV5.KeysPressed[KeyEvent.VK_2] = false;
            movement = 8;
            state = 1;
        } else if (GDV5.KeysPressed[KeyEvent.VK_3]) {
            GDV5.KeysPressed[KeyEvent.VK_3] = false;
            movement = 5;
            state = 1;
        }
    }

    private void handleInstructionsPage() {
        if (GDV5.KeysPressed[KeyEvent.VK_ENTER]) {
            GDV5.KeysPressed[KeyEvent.VK_ENTER] = false;
            resetGame();
            state = 2;
            Sound.playBackgroundMusic();
        } else if (GDV5.KeysPressed[KeyEvent.VK_H]) {
            GDV5.KeysPressed[KeyEvent.VK_H] = false;
            state = 0;
        }
    }

    private void handleGameplay() {
        count++;
        snake.turn(board);

        if (GDV5.KeysPressed[KeyEvent.VK_SPACE] && movement > 2) {
            movement--;
        }

        if (GDV5.KeysPressed[KeyEvent.VK_O]) {
            movement = dMovement;
        }

        if (count % movement == 0) {
            snake.move(board);
        }

        if (snake.snakeHitWall(board)) {
            snakeDead = true;
            state = 3;
            Sound.playGameOverSound();
            Sound.stopBackgroundMusic();
        }

        if (powerUp.eaten(snake)) {
            Sound.playEatSound();
            powerUp.effect(snake);
            powerUp.reset();
        }
    }

    private void handleGameOver() {
        if (GDV5.KeysPressed[KeyEvent.VK_R]) {
            GDV5.KeysPressed[KeyEvent.VK_R] = false;
            resetGame();
        }

        if (GDV5.KeysPressed[KeyEvent.VK_H]) {
            GDV5.KeysPressed[KeyEvent.VK_H] = false;
            oriGame();
            state = 0;
        }
    }

    private void oriGame() {
        state = 2;
        board = new Board();
        snake = new Snake();
        powerUp = new powerUps(20);
        snakeDead = false;
        count = 0;
        state = 0;
        movement = dMovement;
    }

    private void resetGame() {
        state = 2;
        snakeDead = false;
        count = 0;
        movement = dMovement;
        board = new Board();
        snake = new Snake();
        powerUp = new powerUps(20);

        for (int i = 0; i < GDV5.KeysPressed.length; i++) {
            GDV5.KeysPressed[i] = false;
        }

        Sound.stopBackgroundMusic();
        Sound.playBackgroundMusic();
    }

    @Override
    public void draw(Graphics2D win) {
        if (state == 0) {
            drawIntroPage(win);
        } else if (state == 1) {
            drawInstructionsPage(win);
        } else if (state == 2) {
            board.draw(win);
            snake.draw(win);
            powerUp.draw(win);
            drawScore(win);
        } else if (state == 3) {
            drawGameOver(win);
        }
    }

    private void drawScore(Graphics2D win) {
        win.setColor(Color.white);
        win.setFont(new Font("Arial", Font.BOLD, 28));
        win.drawString("Score: " + snake.getScore(), 20, 40);
    }

    private void drawIntroPage(Graphics2D win) {
        win.setColor(Color.WHITE);
        win.setFont(new Font("Roboto", Font.BOLD, 36));
        win.drawString("Vedang Holay Snake", 250, 200);
        win.setFont(new Font("Roboto", Font.BOLD, 28));
        win.drawString("1. Easy", 420, 400);
        win.drawString("2. Medium", 420, 450);
        win.drawString("3. Hard", 420, 500);
    }

    private void drawInstructionsPage(Graphics2D win) {
        win.setColor(new Color(30, 30, 30));
        win.fillRect(0, 0, 1200, 800);

        win.setColor(Color.WHITE);
        win.setFont(new Font("Arial", Font.BOLD, 36));
        win.drawString("Instructions", 450, 100);

        win.setFont(new Font("Arial", Font.BOLD, 28));
        win.drawString("Move the snake with the arrow keys.", 300, 200);
        win.drawString("Try to eat power-ups to gain points and grow.", 300, 250);
        win.drawString("Avoid hitting the walls or yourself.", 300, 300);
        win.drawString("Press Enter to start the game.", 300, 350);
        win.drawString("Press H to return to the home page.", 300, 400);
        win.drawString("Press SPACE to speed up the snake.", 300, 450);
        win.drawString("Press O to reset speed to normal. (4)", 300, 500);

        int startX = 300;
        int startY = 530;
        int size = 30;

        powerUps.drawAppleSprite(win, startX, startY, size, Color.RED);
        win.setColor(Color.WHITE);
        win.drawString("Red: +1 Point & Grow +2", startX + 50, startY + 25);

        powerUps.drawAppleSprite(win, startX, startY + 50, size, Color.GREEN);
        win.setColor(Color.WHITE);
        win.drawString("Green: +5 Points & Grow +2", startX + 50, startY + 75);
        win.drawString("Green bonus: next spawn is 3 red apples", startX + 50, startY + 125);
    }

    private void drawGameOver(Graphics2D win) {
        win.setColor(Color.WHITE);
        win.setFont(new Font("Arial", Font.BOLD, 36));
        if (snake.getScore() >= 50) {
            win.drawString("Game Over, Great Job!", 400, 300);
            win.drawString("Your Final Score: " + snake.getScore(), 400, 360);
        } else {
            win.drawString("Game Over!", 400, 300);
            win.drawString("Final Score: " + snake.getScore(), 400, 360);
        }
        win.drawString("Press R to Restart or H to Return to Home Page", 200, 420);
    }
}

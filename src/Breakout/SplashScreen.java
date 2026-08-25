package Breakout;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.event.KeyEvent;

import game.GDV5;

public class SplashScreen extends GDV5 {

    private boolean gameStarted = false;

    @Override
    public void update() {
        boolean spacePressed = KeysPressed[KeyEvent.VK_SPACE] || KeysTyped[KeyEvent.VK_SPACE];

        if (!gameStarted && spacePressed) {
            gameStarted = true;
            KeysPressed[KeyEvent.VK_SPACE] = false;
            KeysTyped[KeyEvent.VK_SPACE] = false;

            try {
                Gamerunner2 game = new Gamerunner2();
                game.start();
                this.dispose();
            } catch (RuntimeException ex) {
                gameStarted = false;
                ex.printStackTrace();
            }
        }
    }

    @Override
    public void draw(Graphics2D win) {

        win.setColor(Color.BLACK);
        win.fillRect(0, 0, getMaxWindowX(), getMaxWindowY());

        win.setColor(Color.WHITE);
        win.setFont(new Font("Arial", Font.BOLD, 58));
        win.drawString("BREAKOUT", 185, 110);

        win.setColor(Color.WHITE);
        win.setFont(new Font("Arial", Font.BOLD, 26));
        win.drawString("By Vedang Holay", 255, 155);

        win.setColor(Color.BLACK);
        win.fillRoundRect(60, 200, 330, 300, 20, 20);
        win.setColor(Color.WHITE);
        win.setFont(new Font("Arial", Font.BOLD, 24));
        win.drawString("HOW TO PLAY", 130, 240);

        win.setFont(new Font("Arial", Font.PLAIN, 18));
        win.drawString("1. Break all bricks to win.", 85, 280);
        win.drawString("2. Don't let the ball fall.", 85, 315);
        win.drawString("3. Catch pills for boosts:", 85, 350);
        win.setColor(Color.WHITE);
        win.drawString("Yellow: Faster paddle", 105, 385);
        win.drawString("Red: Slower ball", 105, 415);
        win.drawString("Green: Bigger paddle", 105, 445);

        win.setColor(Color.BLACK);
        win.fillRoundRect(420, 200, 300, 300, 20, 20);
        win.setColor(Color.WHITE);
        win.setFont(new Font("Arial", Font.BOLD, 24));
        win.drawString("KEY LAYOUT", 490, 240);

        drawKey(win, "LEFT ARROW", 450, 280);
        drawKey(win, "RIGHT ARROW", 450, 330);
        drawKey(win, "SPACE", 450, 380);
        drawKey(win, "ESC", 450, 430);

        win.setFont(new Font("Arial", Font.PLAIN, 16));
        win.drawString("Move paddle left", 585, 306);
        win.drawString("Move paddle right", 585, 356);
        win.drawString("Start game", 585, 406);
        win.drawString("Quit", 585, 456);
        win.drawString("1. Easy  2. Meidum  3. Hard", 450, 505);

        win.setColor(Color.WHITE);
        win.setFont(new Font("Arial", Font.BOLD, 24));
        win.drawString("Press SPACE to Play", 265, 575);
    }

    private void drawKey(Graphics2D win, String label, int x, int y) {
        win.setColor(Color.BLACK);
        win.fillRoundRect(x, y, 120, 36, 8, 8);
        win.setColor(Color.WHITE);
        win.setFont(new Font("Arial", Font.BOLD, 14));
        win.drawString(label, x + 10, y + 23);
    }

    public static void main(String[] args) {
        SplashScreen splash = new SplashScreen();
        splash.start();
    }
}

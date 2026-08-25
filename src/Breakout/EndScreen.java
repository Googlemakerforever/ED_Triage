package Breakout;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import game.GDV5;

public class EndScreen extends GDV5 {

    private boolean actionTaken = false; 

    @Override
    public void update() {

        
        if (!actionTaken && KeysPressed[10]) { 
            actionTaken = true;

            SplashScreen splash = new SplashScreen();
            splash.start();

            this.dispose();
        }

        
        if (!actionTaken && KeysPressed[27]) { 
            actionTaken = true;

            this.dispose();
            System.exit(0);
        }
    }

    @Override
    public void draw(Graphics2D win) {

        win.setColor(Color.BLACK);
        win.fillRect(0, 0, getMaxWindowX(), getMaxWindowY());

        win.setColor(Color.RED);
        win.setFont(new Font("Arial", Font.BOLD, 60));
        win.drawString("GAME OVER", 170, 300);

        win.setColor(Color.WHITE);
        win.setFont(new Font("Arial", Font.BOLD, 24));
        win.drawString("Press ENTER to Play Again", 180, 380);
        win.drawString("Press ESC to Quit", 235, 420);
    }

    public static void main(String[] args) {
        EndScreen end = new EndScreen();
        end.start();
    }
}

package me.cv.datasetcollect;

import me.voidxwalker.autoreset.Atum;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.TitleScreen;
import net.minecraft.client.options.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.LiteralText;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lwjgl.glfw.GLFW;

public class DatasetCollectMod implements ClientModInitializer {
    public static final String MOD_ID = "datasetcollect";
    public static final Logger LOGGER = LogManager.getLogger(MOD_ID);

    private static KeyBinding startCollectKey;
    private static DatasetCollectConfig config;

    /** Ticks until auto-start; {@code -1} = disabled / already fired. */
    private static int autoStartTicksRemaining = -1;
    private static boolean autoStartFinished;

    @Override
    public void onInitializeClient() {
        DatasetCollector.registerAtumSeedProvider();

        ClientLifecycleEvents.CLIENT_STARTED.register(client -> {
            config = DatasetCollectConfig.load(client);
            DatasetCollector.loadSeeds(client);
            LOGGER.info("Game directory: {}", client.runDirectory.getAbsolutePath());
            LOGGER.info("Expected seeds.txt: {}", client.runDirectory.toPath().resolve("seeds.txt").toAbsolutePath());
            LOGGER.info("Screenshots will go to: {}", DatasetCollector.getDatasetDir(client).toAbsolutePath());

            if (config.autoStart && DatasetCollector.hasSeeds()) {
                autoStartTicksRemaining = config.startDelaySeconds * 20;
                autoStartFinished = false;
                LOGGER.info("Auto-start enabled — collection begins in {} seconds.", config.startDelaySeconds);
            } else if (config.autoStart) {
                LOGGER.warn("Auto-start enabled but seeds.txt is empty — fix seeds and restart, or press F7.");
            }
        });

        startCollectKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.datasetcollect.start",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_F7,
                "key.categories.datasetcollect"
        ));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            tickAutoStart(client);
            tickManualStart(client);
        });

        LOGGER.info("Dataset Collector loaded. Auto-start after delay, or press F7 / check Controls.");
    }

    private static void tickAutoStart(MinecraftClient client) {
        if (autoStartFinished || autoStartTicksRemaining < 0) {
            return;
        }
        if (!config.autoStart || DatasetCollector.isRunning()) {
            autoStartTicksRemaining = -1;
            autoStartFinished = true;
            return;
        }

        if (autoStartTicksRemaining > 0) {
            if (autoStartTicksRemaining % 20 == 0) {
                int secondsLeft = autoStartTicksRemaining / 20;
                LOGGER.info("Auto-start in {} second(s)...", secondsLeft);
            }
            autoStartTicksRemaining--;
            return;
        }

        // Delay elapsed — wait until title screen so we do not interrupt another GUI.
        if (!(client.currentScreen instanceof TitleScreen) && client.world == null) {
            return;
        }
        if (client.world != null) {
            LOGGER.info("Auto-start skipped — already in a world.");
            autoStartFinished = true;
            autoStartTicksRemaining = -1;
            return;
        }

        LOGGER.info("Auto-start triggering collection now.");
        autoStartFinished = true;
        autoStartTicksRemaining = -1;
        beginCollection(client, "auto-start");
    }

    private static boolean f7WasDownLastTick;

    private static void tickManualStart(MinecraftClient client) {
        boolean viaBinding = startCollectKey.wasPressed();
        boolean f7Down = InputUtil.isKeyPressed(client.getWindow().getHandle(), GLFW.GLFW_KEY_F7);
        boolean viaRawKey = f7Down && !f7WasDownLastTick;
        f7WasDownLastTick = f7Down;

        if (!viaBinding && !viaRawKey) {
            return;
        }
        if (DatasetCollector.isRunning()) {
            LOGGER.info("Collection already running.");
            DatasetCollector.notify(client, new LiteralText("Collection already running."));
            return;
        }
        beginCollection(client, viaBinding ? "F7 keybind" : "F7 raw key");
    }

    public static void beginCollection(MinecraftClient client, String reason) {
        if (DatasetCollector.isRunning()) {
            return;
        }
        DatasetCollector.start(client);
        if (!DatasetCollector.isRunning()) {
            return;
        }
        LOGGER.info("Beginning collection ({})", reason);
        DatasetCollector.notify(client, new LiteralText("Starting capture (" + reason + ")"));
        if (Atum.isRunning()) {
            Atum.scheduleReset();
        } else {
            Atum.createNewWorld();
        }
    }
}

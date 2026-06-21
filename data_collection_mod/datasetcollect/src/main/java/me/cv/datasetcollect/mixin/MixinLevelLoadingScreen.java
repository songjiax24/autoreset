package me.cv.datasetcollect.mixin;

import me.cv.datasetcollect.DatasetCollector;
import me.voidxwalker.worldpreview.WorldPreview;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.LevelLoadingScreen;
import net.minecraft.client.util.math.MatrixStack;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Backup frame hook on the vanilla loading screen (World Preview draws on top of this).
 */
@Mixin(LevelLoadingScreen.class)
public abstract class MixinLevelLoadingScreen {

    @Inject(method = "render", at = @At("TAIL"))
    private void datasetcollect$onLoadingScreenFrame(
            MatrixStack matrices,
            int mouseX,
            int mouseY,
            float delta,
            CallbackInfo ci
    ) {
        if (WorldPreview.inPreview()) {
            DatasetCollector.onPreviewFrameRendered(MinecraftClient.getInstance());
        }
    }
}

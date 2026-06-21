package me.cv.datasetcollect.mixin;

import me.cv.datasetcollect.DatasetCollector;
import me.voidxwalker.worldpreview.WorldPreview;
import me.voidxwalker.worldpreview.WorldPreviewProperties;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.widget.ButtonWidget;
import net.minecraft.client.util.math.MatrixStack;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyVariable;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.List;

/**
 * Locks client settings and hides the preview menu overlay during collection.
 */
@Mixin(value = WorldPreviewProperties.class, remap = false)
public abstract class MixinWorldPreview {

    @Inject(method = "render", at = @At("HEAD"), remap = false)
    private void datasetcollect$lockSettings(
            MatrixStack matrices,
            int mouseX,
            int mouseY,
            float delta,
            List<ButtonWidget> buttons,
            int width,
            int height,
            boolean showMenu,
            CallbackInfo ci
    ) {
        if (DatasetCollector.isRunning() && WorldPreview.inPreview()) {
            DatasetCollector.applyClientSettings(MinecraftClient.getInstance());
        }
    }

    @ModifyVariable(
            method = "render",
            at = @At("HEAD"),
            ordinal = 0,
            argsOnly = true,
            remap = false
    )
    private boolean datasetcollect$hideMenuOverlay(boolean showMenu) {
        if (DatasetCollector.isRunning() && WorldPreview.inPreview()) {
            return false;
        }
        return showMenu;
    }
}

import { app } from "../../scripts/app.js";

const TARGET_CLASS = "ArataGemmaShotJsonExport";

function triggerDownload(fileInfo) {
    if (!fileInfo?.relative_output_path) {
        return;
    }

    const anchor = document.createElement("a");
    anchor.href = `/arata-gemma-shots/download?path=${encodeURIComponent(fileInfo.relative_output_path)}`;
    anchor.download = fileInfo.filename || "gemma_shots.json";
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
}

function updateButton(widget, fileInfo, emptyLabel) {
    if (widget) {
        widget.name = fileInfo ? `Download ${fileInfo.filename}` : emptyLabel;
    }
}

app.registerExtension({
    name: "arata.gemma_shots.download_buttons",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_CLASS && nodeType.comfyClass !== TARGET_CLASS) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this._arataGemmaDescriptionFiles = [];

            this._arataGemmaJsonButton = this.addWidget("button", "Download Gemma shots JSON", null, () => {
                const fileInfo = this._arataGemmaDescriptionFiles.find((item) => item.label === "descriptions");
                triggerDownload(fileInfo);
            });

            this._arataGemmaStatusWidget = this.addWidget(
                "text",
                "export_status",
                "Run the node to generate the Gemma descriptions JSON file.",
                () => {}
            );
            this.size = [Math.max(this.size[0], 360), this.size[1]];
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const files = Array.isArray(message?.description_files)
                ? message.description_files
                : Array.isArray(message?.ui?.description_files)
                    ? message.ui.description_files
                    : [];
            this._arataGemmaDescriptionFiles = files;

            const descriptionsFile = files.find((item) => item.label === "descriptions");
            updateButton(this._arataGemmaJsonButton, descriptionsFile, "Download Gemma shots JSON");

            if (this._arataGemmaStatusWidget) {
                this._arataGemmaStatusWidget.value = descriptionsFile
                    ? `Ready: ${descriptionsFile.filename}`
                    : "No Gemma descriptions file was exported.";
            }
            this.setDirtyCanvas(true, true);
        };
    },
});

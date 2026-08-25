import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "h3_open_writer.open_guide_folder",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniMaxH3OpenWriterRef") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            this.addWidget("button", "Open guide folder", null, async () => {
                try {
                    const response = await fetch("/h3_open_writer/open_guide_folder", { method: "POST" });
                    const data = await response.json();
                    if (!data.ok) {
                        alert("Could not open guide folder:\n" + (data.error || "unknown error") + "\n" + (data.path || ""));
                    }
                } catch (error) {
                    alert("Could not open guide folder:\n" + error);
                }
            });
            return result;
        };
    },
});

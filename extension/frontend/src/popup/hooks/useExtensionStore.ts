import browser from "webextension-polyfill";
import {
    getStorageOrDefaults,
    StorageDefaults,
    StorageKeys,
} from "../../shared/storage";
import { create } from "zustand";

export type ExtensionStoreState = {
    serverAddress: string;
    batchSize: number;
    batchTimeout: number;
    apiKey: string;
    loading: boolean;
};

export type ExtensionStoreActions = {
    setServerAddress: (value: string) => void;
    setBatchSize: (value: number) => void;
    setBatchTimeout: (value: number) => void;
    setApiKey: (value: string) => void;
};

export const useExtensionStore = create<
    ExtensionStoreState & ExtensionStoreActions
>((set) => {
    getStorageOrDefaults().then((data) => {
        set({
            serverAddress: data[StorageKeys.ServerAddress],
            batchSize: data[StorageKeys.BatchSize],
            batchTimeout: data[StorageKeys.BatchTimeout],
            apiKey: data[StorageKeys.ApiKey],
            loading: false,
        });
    });

    return {
        loading: true,
        serverAddress: StorageDefaults[StorageKeys.ServerAddress],
        batchSize: StorageDefaults[StorageKeys.BatchSize],
        batchTimeout: StorageDefaults[StorageKeys.BatchTimeout],
        apiKey: StorageDefaults[StorageKeys.ApiKey],
        setServerAddress: (v) => {
            browser.storage.local.set({ [StorageKeys.ServerAddress]: v });
            set({ serverAddress: v });
        },
        setBatchSize: (v) => {
            browser.storage.local.set({ [StorageKeys.BatchSize]: v });
            set({ batchSize: v });
        },
        setBatchTimeout: (v) => {
            browser.storage.local.set({ [StorageKeys.BatchTimeout]: v });
            set({ batchTimeout: v });
        },
        setApiKey: (v) => {
            browser.storage.local.set({ [StorageKeys.ApiKey]: v });
            set({ apiKey: v });
        },
    };
});
